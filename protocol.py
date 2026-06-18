from __future__ import annotations

from dataclasses import dataclass

STATE_SIZE = 3
ACTION_SIZE = 3

ACTION_TO_FORCE: dict[int, float] = {
    0: -1.0,
    1: 0.0,
    2: 1.0,
}


@dataclass
class AgentState:
    id: int
    is_done: bool
    score: int
    has_obstacle: bool
    gap_x: float
    gap_z: float
    vx: float

    def to_vector(self) -> list[float]:
        return [self.gap_x, self.gap_z, self.vx]


@dataclass
class RequestData:
    episode: int
    population: int
    is_done: bool
    agent: AgentState | None

    @classmethod
    def from_json(cls, payload: dict) -> RequestData:
        if "Agents" in payload:
            return cls._from_legacy_json(payload)

        agent = AgentState(
            id=int(payload.get("AgentID", 0)),
            is_done=bool(payload.get("AgentIsDone", True)),
            score=int(payload.get("Score", 0)),
            has_obstacle=bool(payload.get("HasObstacle", False)),
            gap_x=float(payload.get("GapX", 0.0)),
            gap_z=float(payload.get("GapZ", 1.0)),
            vx=float(payload.get("Vx", 0.0)),
        )
        return cls(
            episode=int(payload["Episode"]),
            population=int(payload["Population"]),
            is_done=bool(payload["IsDone"]),
            agent=agent,
        )

    @classmethod
    def _from_legacy_json(cls, payload: dict) -> RequestData:
        agents = payload.get("Agents") or []
        if not agents:
            return cls(
                episode=int(payload["Episode"]),
                population=int(payload["Population"]),
                is_done=bool(payload["IsDone"]),
                agent=None,
            )
        raw = agents[0]
        agent = AgentState(
            id=int(raw["ID"]),
            is_done=bool(raw["IsDone"]),
            score=int(raw["Score"]),
            has_obstacle=bool(raw.get("HasObstacle", False)),
            gap_x=float(raw.get("GapX", 0.0)),
            gap_z=float(raw.get("GapZ", 1.0)),
            vx=float(raw.get("Vx", 0.0)),
        )
        return cls(
            episode=int(payload["Episode"]),
            population=int(payload["Population"]),
            is_done=bool(payload["IsDone"]),
            agent=agent,
        )


@dataclass
class ResponseData:
    force_x: float
    is_done: bool = False

    def to_json_dict(self) -> dict:
        return {
            "ForceX": self.force_x,
            "IsDone": self.is_done,
        }


def action_to_force(action: int) -> float:
    return ACTION_TO_FORCE.get(action, 0.0)
