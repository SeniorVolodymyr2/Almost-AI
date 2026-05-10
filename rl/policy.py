from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        action_scale: float = 1.0,
        init_log_std: float = -1.5,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.tensor(init_log_std))

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs) * self.action_scale
        std = torch.exp(self.log_std).clamp(min=1e-4, max=2.0)
        return Normal(mean, std)
