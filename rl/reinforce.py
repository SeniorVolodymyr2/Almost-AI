from __future__ import annotations

import logging
import os
import socket
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.optim as optim

from config import (
    DIAG_LOG_EVERY_N_STEPS,
    ELITE_WEIGHT,
    ENTROPY_COEF_END,
    ENTROPY_COEF_START,
    ENTROPY_DECAY_STEPS,
    FORCE_X_CLIP,
    GAMMA,
    GAP_SHAPING_ENABLED,
    GAP_SHAPING_W_X,
    GAP_SHAPING_W_Z,
    HOST,
    LEARNING_RATE,
    MODELS_DIR,
    MULTI_AGENT_AVG_TRAJECTORIES,
    NORMALIZE_RETURNS,
    PORT,
    SAVE_EVERY_EPISODES,
)
from rl.observation import encode_agent, observation_dim_from_agent
from rl.policy import GaussianPolicy
from serializers import (
    AgentAction,
    AgentState,
    RequestData,
    ResponseData,
    deserialize_request,
    serialize_response,
)

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    obs: torch.Tensor
    action: torch.Tensor


@dataclass
class AgentRollout:
    steps: list[StepRecord] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


@dataclass
class RolloutLoss:
    agent_id: int
    pg_loss: torch.Tensor
    entropy_sum: torch.Tensor
    n_steps: int
    total_return: float

    def combined_loss(self, entropy_coef: float) -> torch.Tensor:
        return self.pg_loss - entropy_coef * self.entropy_sum


def _potential_gap(gap_dx: float, gap_dz: float) -> float:
    return -GAP_SHAPING_W_X * abs(gap_dx) - GAP_SHAPING_W_Z * abs(gap_dz)


def shaping_reward_gap(prev_dx: float, prev_dz: float, curr_dx: float, curr_dz: float) -> float:
    if not GAP_SHAPING_ENABLED:
        return 0.0
    phi_s = _potential_gap(prev_dx, prev_dz)
    phi_sp = _potential_gap(curr_dx, curr_dz)
    return GAMMA * phi_sp - phi_s


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    g = 0.0
    out: list[float] = []
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return out


class ReinforceClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.host = host if host is not None else HOST
        self.port = port if port is not None else PORT
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy: Optional[GaussianPolicy] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.obs_dim: Optional[int] = None

        self._last_request_episode: Optional[int] = None
        self._completed_episodes = 0
        self._total_trajectory_updates = 0
        self._optimizer_steps = 0

        self._pending: dict[int, tuple[torch.Tensor, torch.Tensor, float, float, float]] = {}
        self._rollout: dict[int, AgentRollout] = {}

        self._diag_env_step = 0
        self._batch_r_env_sum = 0.0
        self._batch_r_shape_sum = 0.0
        self._batch_reward_steps = 0

    def _entropy_coef(self) -> float:
        if ENTROPY_DECAY_STEPS <= 0:
            return ENTROPY_COEF_START
        t = min(1.0, float(self._optimizer_steps) / float(ENTROPY_DECAY_STEPS))
        return ENTROPY_COEF_START * (1.0 - t) + ENTROPY_COEF_END * t

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            stream=sys.stdout,
        )
        logger.info(
            "REINFORCE gap_shaping=%s (w_x=%s w_z=%s) → %s:%s",
            GAP_SHAPING_ENABLED,
            GAP_SHAPING_W_X,
            GAP_SHAPING_W_Z,
            self.host,
            self.port,
        )

        os.makedirs(MODELS_DIR, exist_ok=True)

        try:
            with socket.create_connection((self.host, self.port), timeout=120) as sock:
                sock.settimeout(None)
                logger.info("Connected.")
                self._io_loop(sock)
        except ConnectionRefusedError:
            logger.error("Connection refused — is Unity listening on %s:%s?", self.host, self.port)
            raise

    def _ensure_policy(self, sample_agent: AgentState) -> None:
        if self.policy is not None:
            return
        dim = observation_dim_from_agent(sample_agent)
        self.obs_dim = dim
        self.policy = GaussianPolicy(
            obs_dim=dim,
            hidden_dim=128,
            action_scale=float(FORCE_X_CLIP),
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        logger.info(
            "GaussianPolicy obs_dim=%s device=%s entropy %s→%s over %s steps",
            dim,
            self.device,
            ENTROPY_COEF_START,
            ENTROPY_COEF_END,
            ENTROPY_DECAY_STEPS,
        )

    def _io_loop(self, sock: socket.socket) -> None:
        buffer = ""
        while True:
            data = sock.recv(65536)
            if not data:
                logger.info("Unity closed connection.")
                break

            buffer += data.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                try:
                    request = deserialize_request(line)
                except Exception as e:
                    logger.warning("Bad JSON line skipped: %s", e)
                    continue

                alive = [a for a in request.Agents if not a.IsDone]
                if alive:
                    self._ensure_policy(alive[0])

                response = self._step(request)
                sock.sendall(serialize_response(response).encode("utf-8"))

                if response.IsDone:
                    logger.info("Sent training finished to Unity; exiting client.")
                    return

    def _step(self, request: RequestData) -> ResponseData:
        assert self.policy is not None and self.optimizer is not None

        if self._last_request_episode is None or request.Episode != self._last_request_episode:
            if self._last_request_episode is not None:
                logger.info("Unity Episode → %s", request.Episode)
            self._last_request_episode = request.Episode
            self._pending.clear()
            self._rollout.clear()

        self._batch_r_env_sum = 0.0
        self._batch_r_shape_sum = 0.0
        self._batch_reward_steps = 0

        actions_out: list[AgentAction] = []
        rollout_losses: list[RolloutLoss] = []

        for agent in request.Agents:
            aid = agent.ID

            if aid in self._pending:
                prev_obs, prev_action, cum_when_acting, prev_dx, prev_dz = self._pending.pop(aid)
                r_env = float(agent.Reward) - cum_when_acting
                r_shape = shaping_reward_gap(
                    prev_dx,
                    prev_dz,
                    float(agent.GapDeltaX),
                    float(agent.GapDeltaZ),
                )
                r = r_env + r_shape

                self._batch_r_env_sum += r_env
                self._batch_r_shape_sum += r_shape
                self._batch_reward_steps += 1

                if aid not in self._rollout:
                    self._rollout[aid] = AgentRollout()
                ro = self._rollout[aid]
                ro.steps.append(StepRecord(obs=prev_obs, action=prev_action))
                ro.rewards.append(r)

                if agent.IsDone:
                    rl = self._finalize_rollout(aid)
                    if rl is not None:
                        rollout_losses.append(rl)

            if agent.IsDone:
                continue

            obs = encode_agent(agent, self.device)
            self.policy.train()
            dist = self.policy(obs)
            action = dist.sample()
            cumulative_reward = float(agent.Reward)
            self._pending[aid] = (
                obs.detach(),
                action.detach(),
                cumulative_reward,
                float(agent.GapDeltaX),
                float(agent.GapDeltaZ),
            )

            force_sent = float(torch.clamp(action, -FORCE_X_CLIP, FORCE_X_CLIP).item())
            actions_out.append(AgentAction(ID=aid, ForceX=force_sent))

        if request.IsDone:
            for aid in list(self._rollout.keys()):
                if self._rollout[aid].steps:
                    rl = self._finalize_rollout(aid)
                    if rl is not None:
                        rollout_losses.append(rl)

            self._apply_rollout_losses(rollout_losses, request.Population)

            self._completed_episodes += 1
            alive_x = [float(a.AgentX) for a in request.Agents if not a.IsDone]
            if alive_x:
                logger.info(
                    "episode_done episode=%s pop=%s mean_AgentX_alive=%.4f",
                    self._completed_episodes,
                    request.Population,
                    statistics.mean(alive_x),
                )
            else:
                logger.info(
                    "episode_done episode=%s pop=%s (no survivors last frame)",
                    self._completed_episodes,
                    request.Population,
                )
            self._save_checkpoint_maybe()

            return ResponseData(Agents=[], IsDone=False)

        self._apply_rollout_losses(rollout_losses, request.Population)

        self._maybe_diag_actions(request, actions_out)

        return ResponseData(Agents=actions_out, IsDone=False)

    def _maybe_diag_actions(
        self,
        request: RequestData,
        actions_out: list[AgentAction],
    ) -> None:
        if not actions_out or DIAG_LOG_EVERY_N_STEPS <= 0:
            return
        self._diag_env_step += 1
        if self._diag_env_step % DIAG_LOG_EVERY_N_STEPS != 0:
            return

        id_to_x = {a.ID: float(a.AgentX) for a in request.Agents}
        fxs = [float(a.ForceX) for a in actions_out]
        xs = [id_to_x.get(a.ID, 0.0) for a in actions_out]
        std_f = statistics.pstdev(fxs) if len(fxs) > 1 else 0.0
        std_x = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        mean_env = mean_shape = 0.0
        if self._batch_reward_steps > 0:
            mean_env = self._batch_r_env_sum / self._batch_reward_steps
            mean_shape = self._batch_r_shape_sum / self._batch_reward_steps
        logger.info(
            "diag env_step=%s mean_ForceX=%.4f std_ForceX=%.4f mean_AgentX=%.4f std_AgentX=%.4f n=%s entropy_coef=%.5f "
            "mean_r_env=%.5f mean_r_shape=%.5f reward_steps=%s",
            self._diag_env_step,
            statistics.mean(fxs),
            std_f,
            statistics.mean(xs),
            std_x,
            len(actions_out),
            self._entropy_coef(),
            mean_env,
            mean_shape,
            self._batch_reward_steps,
        )

    def _finalize_rollout(self, aid: int) -> Optional[RolloutLoss]:
        ro = self._rollout.pop(aid, None)
        if ro is None or not ro.steps:
            return None
        if len(ro.steps) != len(ro.rewards):
            logger.warning("Trajectory length mismatch agent %s", aid)
            return None
        return self._policy_loss_from_rollout(aid, ro)

    def _policy_loss_from_rollout(self, aid: int, ro: AgentRollout) -> RolloutLoss:
        returns = discounted_returns(ro.rewards, GAMMA)
        if NORMALIZE_RETURNS and len(returns) > 1:
            t_ret = torch.tensor(returns, device=self.device, dtype=torch.float32)
            t_ret = (t_ret - t_ret.mean()) / (t_ret.std() + 1e-8)
        else:
            t_ret = torch.tensor(returns, device=self.device, dtype=torch.float32)

        self.policy.train()
        pg = torch.zeros((), device=self.device)
        ent_sum = torch.zeros((), device=self.device)
        for rec, g in zip(ro.steps, t_ret):
            dist = self.policy(rec.obs)
            logp = dist.log_prob(rec.action).sum()
            pg -= logp * g
            ent_sum += dist.entropy().sum()

        total_return = float(sum(ro.rewards))
        self._total_trajectory_updates += 1
        return RolloutLoss(
            agent_id=aid,
            pg_loss=pg,
            entropy_sum=ent_sum,
            n_steps=len(ro.steps),
            total_return=total_return,
        )

    def _apply_rollout_losses(
        self,
        rollout_losses: list[RolloutLoss],
        population_hint: int,
    ) -> None:
        if not rollout_losses:
            return

        beta = self._entropy_coef()
        max_ret = max(r.total_return for r in rollout_losses)

        weighted_terms: list[torch.Tensor] = []
        weights: list[float] = []
        for r in rollout_losses:
            combined = r.combined_loss(beta)
            w = ELITE_WEIGHT if r.total_return >= max_ret - 1e-9 else 1.0
            weighted_terms.append(combined * w)
            weights.append(w)

        total = torch.stack(weighted_terms).sum()
        if MULTI_AGENT_AVG_TRAJECTORIES:
            total = total / float(sum(weights))

        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self._optimizer_steps += 1

        elite_ids = [r.agent_id for r in rollout_losses if r.total_return >= max_ret - 1e-9]
        logger.info(
            "REINFORCE traj=%s pop=%s loss=%.4f beta=%.5f elite_ids=%s max_ret=%.4f weights_sum=%.2f opt_step=%s",
            len(rollout_losses),
            population_hint,
            float(total.item()),
            beta,
            elite_ids,
            max_ret,
            sum(weights),
            self._optimizer_steps,
        )

    def _save_checkpoint_maybe(self) -> None:
        if self.policy is None:
            return
        if self._completed_episodes > 0 and self._completed_episodes % SAVE_EVERY_EPISODES == 0:
            path = os.path.join(MODELS_DIR, f"reinforce_ep{self._completed_episodes}.pt")
            torch.save(
                {
                    "policy": self.policy.state_dict(),
                    "obs_dim": self.obs_dim,
                    "episodes": self._completed_episodes,
                    "optimizer_steps": self._optimizer_steps,
                },
                path,
            )
            logger.info("Saved %s", path)
