"""Quick confidence checks for official grading behavior."""

from models import CircuitState
from server.grader import (
    SUCCESS_SCORE_THRESHOLD,
    SUCCESS_TOLERANCE,
    WEIGHT_ACCURACY,
    WEIGHT_COST,
    WEIGHT_STEP,
    clamp_score,
    grade_episode,
    grade_task_result,
    is_success,
    normalized_score,
)


def test_clamp_score_bounds():
    assert clamp_score(-0.1) == 0.0
    assert clamp_score(1.2) == 1.0


def test_grade_episode_returns_best_score():
    state = CircuitState(
        task_id="lp_1khz_budget",
        circuit_type="low_pass",
        target_hz=1000.0,
        step_count=3,
        cumulative_reward=1.7,
        best_score=0.83,
        done=True,
    )
    assert grade_episode(state) == 0.83


def test_is_success_uses_threshold():
    assert is_success(0.81) is True
    assert is_success(0.79) is False
    assert SUCCESS_SCORE_THRESHOLD == 0.8


def test_grade_task_result_includes_expected_fields():
    state = CircuitState(
        task_id="lp_1khz_budget",
        circuit_type="low_pass",
        target_hz=1000.0,
        step_count=5,
        cumulative_reward=2.3,
        best_score=0.9,
        done=True,
    )
    result = grade_task_result(state)

    assert result["task_id"] == "lp_1khz_budget"
    assert result["score"] == 0.9
    assert result["success"] is True
    assert result["steps"] == 5
    assert result["best_score"] == 0.9
    assert result["done"] is True


def test_normalized_score_bounds():
    assert 0.0 <= normalized_score(0.0, 0.0, 0, 8) <= 1.0
    assert 0.0 <= normalized_score(1.0, 1.0, 8, 8) <= 1.0


def test_frozen_weights_and_tolerance():
    assert WEIGHT_ACCURACY == 0.7
    assert WEIGHT_COST == 0.2
    assert WEIGHT_STEP == 0.1
    assert SUCCESS_TOLERANCE == 0.02
