from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from baseline import rule_based_action
from env import RacingEnv
from model_paths import episode_model_path, find_latest_model, resolve_load_path
from protocol import ACTION_SIZE, AgentState, RequestData, ResponseData, action_to_force
from replay_buffer import ReplayBuffer, Transition
from tracker import MetricsTracker
from train_dqn import DqnConfig, DqnTrainer


class RacingBrain:
    def __init__(
        self,
        train: bool,
        use_baseline: bool,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        model_name: str,
        models_dir: str,
        load_episode: int | None,
        load_latest: bool,
        max_episodes: int,
        save_every_episodes: int,
        evaluation_dir: str,
        train_every_n_steps: int,
    ) -> None:
        self.train_enabled = train
        self.use_baseline = use_baseline
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model_name = model_name
        self.models_dir = models_dir
        self.max_episodes = max_episodes
        self.save_every_episodes = save_every_episodes
        self.train_every_n_steps = max(1, train_every_n_steps)
        self.metrics = MetricsTracker(evaluation_dir=evaluation_dir, model_name=model_name)

        self.env = RacingEnv()
        self.replay_buffer = ReplayBuffer(capacity=100_000)
        self.trainer = DqnTrainer(DqnConfig())
        self.last_state: list[float] | None = None
        self.last_action: int | None = None
        self.step_count = 0
        self.completed_episodes = 0
        self.last_unity_episode = 1
        self._state_lock = threading.Lock()
        self._train_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="racing-train")

        load_path = self._resolve_load_path(load_episode, load_latest)
        if load_path is not None:
            try:
                state_dict = torch.load(load_path, map_location=self.trainer.device)
                self.trainer.policy_net.load_state_dict(state_dict)
                self.trainer.target_net.load_state_dict(state_dict)
                print(f"Loaded model from {load_path}")
            except Exception as exc:
                print(f"Failed to load {load_path}: {exc}")
        else:
            print("Starting with fresh weights")

    def _resolve_load_path(self, load_episode: int | None, load_latest: bool):
        if load_episode is not None:
            return resolve_load_path(self.model_name, self.models_dir, load_episode)
        if load_latest:
            return find_latest_model(self.model_name, self.models_dir)
        return None

    def shutdown(self) -> None:
        self._train_executor.shutdown(wait=False, cancel_futures=True)

    def save_model(self, episode: int | None = None) -> None:
        with self._state_lock:
            self._save_model_unlocked(episode)

    def _save_model_unlocked(self, episode: int | None = None) -> None:
        episode_number = episode if episode is not None else self.last_unity_episode
        path = episode_model_path(self.model_name, episode_number, self.models_dir)
        torch.save(self.trainer.policy_net.state_dict(), path)

    def handle_request(self, request: RequestData) -> ResponseData:
        with self._state_lock:
            return self._handle_request_locked(request)

    def _handle_request_locked(self, request: RequestData) -> ResponseData:
        self.last_unity_episode = request.episode
        self.env.reset_episode(request.episode)

        if request.agent is None:
            return ResponseData(force_x=0.0, is_done=self._should_stop(request.episode))

        agent = request.agent
        step = self.env.compute_step(agent, request.is_done)
        state = agent.to_vector()

        if self.last_state is not None and self.last_action is not None:
            transition = Transition(
                state=self.last_state,
                action=self.last_action,
                reward=step.reward,
                next_state=state,
                done=step.done,
            )
            self.replay_buffer.add(transition)
            if self.train_enabled and self.step_count % self.train_every_n_steps == 0:
                self._train_executor.submit(self._train_once)

        self.step_count += 1

        if request.is_done and agent.is_done:
            self._record_episode(agent, request.episode)
            self.last_state = None
            self.last_action = None
            action = 1
        elif agent.is_done:
            action = 1
        else:
            action = self._choose_action(agent, state)
            self.last_state = state
            self.last_action = action

        response = ResponseData(
            force_x=action_to_force(action),
            is_done=self._should_stop(request.episode),
        )

        if response.is_done and self.train_enabled:
            self._save_model_unlocked(request.episode)

        return response

    def _should_stop(self, episode: int) -> bool:
        return self.train_enabled and episode >= self.max_episodes

    def _choose_action(self, agent: AgentState, state: list[float]) -> int:
        if self.use_baseline:
            return rule_based_action(agent)

        if self.train_enabled and random.random() < self.epsilon:
            return self._explore_action(agent)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.trainer.device).unsqueeze(0)
            q_values = self.trainer.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def _explore_action(self, agent: AgentState) -> int:
        """Epsilon-random with bias toward the gap — helps early learning."""
        if agent.has_obstacle and abs(agent.gap_x) > 0.05 and random.random() < 0.8:
            return 2 if agent.gap_x > 0 else 0
        return random.randint(0, ACTION_SIZE - 1)

    def _train_once(self) -> None:
        with self._state_lock:
            self.trainer.train_step(self.replay_buffer)

    def _record_episode(self, agent: AgentState, unity_episode: int) -> None:
        self.completed_episodes += 1
        self._decay_epsilon()
        total_reward = self.env.episode_reward

        record = self.metrics.record_episode(
            unity_episode=unity_episode,
            score=agent.score,
            total_reward=total_reward,
            epsilon=self.epsilon,
        )
        chart_path = self.metrics.update_chart()

        print(
            f"ep {record.episode} score={agent.score} reward={total_reward:.2f} "
            f"eps={self.epsilon:.2f} buf={len(self.replay_buffer)} -> {chart_path.name}"
        )

        if self.train_enabled and self.completed_episodes % self.save_every_episodes == 0:
            self._save_model_unlocked(unity_episode)

    def _decay_epsilon(self) -> None:
        if self.train_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
