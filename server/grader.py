"""Normalized scoring utilities that map performance to [0, 1]."""

from __future__ import annotations


def normalized_score(frequency_error_ratio: float, over_budget_ratio: float) -> float:
    """Combine frequency and budget penalties into a bounded score.

    A score of 1 is perfect, and 0 is worst-case after clipping.
    """

    freq_penalty = min(max(frequency_error_ratio, 0.0), 1.0)
    budget_penalty = min(max(over_budget_ratio, 0.0), 1.0)
    score = 1.0 - (0.7 * freq_penalty + 0.3 * budget_penalty)
    return max(0.0, min(1.0, score))
