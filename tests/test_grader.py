"""Quick confidence checks for normalized grading behavior."""

from server.grader import normalized_score


def test_normalized_score_bounds():
    assert 0.0 <= normalized_score(0.0, 0.0, 0, 8, 0.2, 0.1) <= 1.0
    assert 0.0 <= normalized_score(1.0, 1.0, 8, 8, 0.2, 0.1) <= 1.0


def test_perfect_score_is_one():
    assert normalized_score(0.0, 0.0, 0, 8, 0.2, 0.1) == 1.0
