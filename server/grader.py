"""Official benchmark scoring/reporting helpers."""

from __future__ import annotations

from models import CircuitState
from server.simulator import SUCCESS_TOLERANCE, WEIGHT_ACCURACY, WEIGHT_COST, WEIGHT_STEP, clamp_value, compute_reward

SUCCESS_SCORE_THRESHOLD = 0.8


def clamp_score(score: float) -> float:
    """Clamp any reported score into the benchmark range [0, 1]."""

    return clamp_value(score, 0.0, 1.0)


def grade_episode(state: CircuitState) -> float:
    """Return the frozen final score for one episode."""

    return clamp_score(state.best_score)


def is_success(score: float, threshold: float = SUCCESS_SCORE_THRESHOLD) -> bool:
    """Return whether a score clears the benchmark success threshold."""

    return clamp_score(score) >= threshold


def grade_task_result(state: CircuitState) -> dict[str, float | bool | int | str]:
    """Return a compact benchmark-friendly grading payload."""

    score = grade_episode(state)
    return {
        "task_id": state.task_id,
        "score": score,
        "success": is_success(score),
        "steps": state.step_count,
        "best_score": score,
        "done": state.done,
    }


def normalized_score(
    normalized_error: float,
    normalized_cost: float,
    steps_used: int,
    max_steps: int,
) -> float:
    """Backward-compatible reward wrapper used by existing runtime tests."""

    safe_target = 1.0
    current_hz = safe_target * (1.0 - normalized_error)
    return compute_reward(current_hz, safe_target, normalized_cost, steps_used, max_steps)
