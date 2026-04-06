"""Normalized scoring utilities that map performance to [0, 1]."""

from __future__ import annotations

WEIGHT_ACCURACY = 0.7
WEIGHT_COST = 0.2
WEIGHT_STEP = 0.1
SUCCESS_TOLERANCE = 0.02


def normalized_score(
    normalized_error: float,
    normalized_cost: float,
    steps_used: int,
    max_steps: int,
) -> float:
    """Combine frozen accuracy, cost, and step terms into a bounded reward.

    A score of 1 is perfect, and 0 is worst-case after clipping.
    """

    accuracy_score = 1.0 - min(max(normalized_error, 0.0), 1.0)
    cost_efficiency = 1.0 - min(max(normalized_cost, 0.0), 1.0)
    step_ratio = min(max(steps_used / max(max_steps, 1), 0.0), 1.0)
    step_efficiency = 1.0 - step_ratio
    score = (
        WEIGHT_ACCURACY * accuracy_score
        + WEIGHT_COST * cost_efficiency
        + WEIGHT_STEP * step_efficiency
    )
    if abs(score - 1.0) <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, score))
