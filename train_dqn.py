from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import QNetwork
from protocol import ACTION_SIZE, STATE_SIZE
from replay_buffer import ReplayBuffer


@dataclass
class DqnConfig:
    state_size: int = STATE_SIZE
    action_size: int = ACTION_SIZE
    gamma: float = 0.99
    batch_size: int = 128
    learning_rate: float = 1e-3
    tau: float = 0.01
    min_buffer_size: int = 200
    grad_steps_per_update: int = 2
    grad_clip_norm: float = 10.0


class DqnTrainer:
    def __init__(self, config: DqnConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(config.state_size, config.action_size).to(self.device)
        self.target_net = QNetwork(config.state_size, config.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

    def train_step(self, replay_buffer: ReplayBuffer) -> float | None:
        if len(replay_buffer) < max(self.config.batch_size, self.config.min_buffer_size):
            return None

        total_loss = 0.0
        steps = 0
        for _ in range(self.config.grad_steps_per_update):
            loss = self._single_grad_step(replay_buffer)
            if loss is None:
                break
            total_loss += loss
            steps += 1

        if steps == 0:
            return None
        return total_loss / steps

    def _single_grad_step(self, replay_buffer: ReplayBuffer) -> float | None:
        if len(replay_buffer) < self.config.batch_size:
            return None

        batch = replay_buffer.sample(self.config.batch_size)
        state = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        action = torch.tensor([item.action for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor([item.next_state for item in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)

        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            target_q_values = reward + (1.0 - done) * self.config.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self._soft_update_target()
        return float(loss.item())

    def _soft_update_target(self) -> None:
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.config.tau) * target_param.data + self.config.tau * policy_param.data
            )
