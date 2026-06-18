from __future__ import annotations

from protocol import AgentState


def rule_based_action(agent: AgentState) -> int:
    """Steer toward the gap when an obstacle is visible."""
    if agent.is_done or not agent.has_obstacle:
        return 1
    if abs(agent.gap_x) <= 0.05:
        return 1
    return 2 if agent.gap_x > 0 else 0
