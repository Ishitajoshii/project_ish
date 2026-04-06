"""Quick confidence checks for normalized grading behavior."""

from server.grader import SUCCESS_TOLERANCE, WEIGHT_ACCURACY, WEIGHT_COST, WEIGHT_STEP, normalized_score


def test_normalized_score_bounds():
    assert 0.0 <= normalized_score(0.0, 0.0, 0, 8) <= 1.0
    assert 0.0 <= normalized_score(1.0, 1.0, 8, 8) <= 1.0


def test_perfect_score_is_one():
    assert normalized_score(0.0, 0.0, 0, 8) == 1.0


def test_frozen_weights_and_tolerance():
    assert WEIGHT_ACCURACY == 0.7
    assert WEIGHT_COST == 0.2
    assert WEIGHT_STEP == 0.1
    assert SUCCESS_TOLERANCE == 0.02
