from __future__ import annotations

import torch

from serializers import AgentState

MAX_SPEED = 15.0
TRACK_SCALE = 12.0
TRACK_DEPTH_SCALE = 50.0


def observation_dim_from_agent(_: AgentState) -> int:
    return 4


def encode_agent(agent: AgentState, device: torch.device) -> torch.Tensor:
    x = agent.AgentX / TRACK_SCALE
    dx = agent.GapDeltaX / TRACK_SCALE
    dz = agent.GapDeltaZ / TRACK_DEPTH_SCALE
    v = agent.CurrentSpeed / MAX_SPEED
    t = torch.tensor([x, dx, dz, v], dtype=torch.float32, device=device).unsqueeze(0)
    return torch.clamp(t, -2.0, 2.0)
