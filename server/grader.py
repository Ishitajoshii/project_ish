"""Normalized scoring utilities that map performance to [0, 1]."""

from __future__ import annotations


def normalized_score(
    normalized_error: float,
    normalized_cost: float,
    steps_used: int,
    max_steps: int,
    cost_weight: float,
    step_weight: float,
) -> float:
    """Combine accuracy, cost, and step efficiency into a bounded score.

    A score of 1 is perfect, and 0 is worst-case after clipping.
    """

    accuracy_weight = max(0.0, 1.0 - cost_weight - step_weight)
    accuracy_score = 1.0 - min(max(normalized_error, 0.0), 1.0)
    cost_efficiency = 1.0 - min(max(normalized_cost, 0.0), 1.0)
    step_ratio = min(max(steps_used / max(max_steps, 1), 0.0), 1.0)
    step_efficiency = 1.0 - step_ratio
    score = (
        accuracy_weight * accuracy_score
        + float(cost_weight) * cost_efficiency
        + float(step_weight) * step_efficiency
    )
    return max(0.0, min(1.0, score))
