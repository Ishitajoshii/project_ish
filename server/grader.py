"""Normalized scoring utilities that map performance to [0, 1]."""

from __future__ import annotations

from server.simulator import SUCCESS_TOLERANCE, WEIGHT_ACCURACY, WEIGHT_COST, WEIGHT_STEP, compute_reward


def normalized_score(
    normalized_error: float,
    normalized_cost: float,
    steps_used: int,
    max_steps: int,
) -> float:
    """Backward-compatible reward wrapper used by tests and scoring callers."""

    safe_target = 1.0
    current_hz = safe_target * (1.0 - normalized_error)
    return compute_reward(current_hz, safe_target, normalized_cost, steps_used, max_steps)
