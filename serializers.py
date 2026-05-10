from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class AgentState:
    ID: int
    IsDone: bool
    Score: int
    Reward: float
    CurrentSpeed: float
    AgentX: float
    GapDeltaX: float
    GapDeltaZ: float


@dataclass
class RequestData:
    Episode: int
    Population: int
    IsDone: bool
    Agents: list[AgentState]


@dataclass
class AgentAction:
    ID: int
    ForceX: float

    def __post_init__(self):
        self.ForceX = round(self.ForceX, 3)


@dataclass
class ResponseData:
    Agents: list[AgentAction]
    IsDone: bool


def deserialize_request(json_str: str) -> RequestData:
    data = json.loads(json_str)
    data["Agents"] = [AgentState(**agent) for agent in data["Agents"]]
    return RequestData(**data)


def serialize_response(response: ResponseData) -> str:
    return json.dumps(asdict(response)) + "\n"
