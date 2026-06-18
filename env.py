from __future__ import annotations

from dataclasses import dataclass

from protocol import AgentState


@dataclass
class StepResult:
    reward: float
    done: bool
    score: int


class RacingEnv:
    """Tracks previous agent state and computes Python-side rewards."""

    def __init__(self) -> None:
        self._prev_agent: AgentState | None = None
        self._episode: int | None = None
        self._episode_reward = 0.0
        self._episode_steps = 0

    def reset_episode(self, episode: int) -> None:
        if self._episode != episode:
            self._episode = episode
            self._episode_reward = 0.0
            self._episode_steps = 0
            self._prev_agent = None

    def compute_step(self, agent: AgentState, episode_done: bool) -> StepResult:
        reward = 0.0
        done = agent.is_done

        if self._prev_agent is not None:
            prev = self._prev_agent

            if agent.is_done and not prev.is_done:
                reward = -1.0
                done = True
            elif agent.score > prev.score:
                reward = 3.0
            elif not agent.is_done and agent.has_obstacle:
                gap_progress = abs(prev.gap_x) - abs(agent.gap_x)
                reward += 0.05 * gap_progress
                if agent.gap_z < 0.6 and abs(agent.gap_x) < 0.1:
                    reward += 0.05

        self._episode_reward += reward
        self._episode_steps += 1
        self._prev_agent = agent

        if episode_done and agent.is_done:
            done = True

        return StepResult(reward=reward, done=done, score=agent.score)

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    @property
    def prev_state(self) -> list[float] | None:
        if self._prev_agent is None:
            return None
        return self._prev_agent.to_vector()
