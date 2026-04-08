"""Quick confidence checks for simulator equations and action updates."""

import pytest

from server.simulator import (
    ACTION_SCALE_FACTOR,
    SUCCESS_TOLERANCE,
    apply_action,
    compute_cutoff_hz,
    compute_normalized_cost,
    compute_normalized_error,
    compute_reward,
    compute_step_efficiency,
    evaluate_circuit_state,
    gain_db,
    is_done,
    normalize_log_value,
)


def test_apply_action_updates_component_multiplicatively():
    new_r, new_c, error = apply_action(
        1000.0,
        1e-7,
        "r_up",
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-10,
        1e-3,
    )
    assert error is None
    assert new_r == 1200.0
    assert new_c == 1e-7


def test_apply_action_c_down_changes_only_c():
    new_r, new_c, error = apply_action(
        1000.0,
        1e-6,
        "c_down",
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-9,
        1e-3,
    )
    assert error is None
    assert new_r == 1000.0
    assert new_c == 1e-6 / ACTION_SCALE_FACTOR


def test_apply_action_r_down_changes_only_r():
    new_r, new_c, error = apply_action(
        1000.0,
        1e-6,
        "r_down",
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-9,
        1e-3,
    )
    assert error is None
    assert new_r == 1000.0 / ACTION_SCALE_FACTOR
    assert new_c == 1e-6


def test_apply_action_c_up_changes_only_c():
    new_r, new_c, error = apply_action(
        1000.0,
        1e-6,
        "c_up",
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-9,
        1e-3,
    )
    assert error is None
    assert new_r == 1000.0
    assert new_c == 1e-6 * ACTION_SCALE_FACTOR


@pytest.mark.parametrize(
    ("start_r", "start_c", "action", "expected_r", "expected_c"),
    [
        (1_000_000.0, 1e-9, "r_up", 1_000_000.0, 1e-9),
        (100.0, 1e-9, "r_down", 100.0, 1e-9),
        (1000.0, 1e-3, "c_up", 1000.0, 1e-3),
        (1000.0, 1e-10, "c_down", 1000.0, 1e-10),
    ],
)
def test_apply_action_clamps_to_bounds(start_r, start_c, action, expected_r, expected_c):
    new_r, new_c, error = apply_action(
        start_r,
        start_c,
        action,
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-10,
        1e-3,
    )
    assert error is None
    assert new_r == expected_r
    assert new_c == expected_c


def test_apply_action_invalid_action_returns_originals_and_error():
    new_r, new_c, error = apply_action(
        1000.0,
        1e-6,
        "foo",
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-9,
        1e-3,
    )
    assert new_r == 1000.0
    assert new_c == 1e-6
    assert error == "invalid action: foo"


def test_compute_cutoff_hz_basic():
    hz = compute_cutoff_hz(1000.0, 1e-6)
    assert 159.0 < hz < 159.2


def test_compute_cutoff_hz_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="r_ohms must be > 0"):
        compute_cutoff_hz(0.0, 1e-6)

    with pytest.raises(ValueError, match="r_ohms must be > 0"):
        compute_cutoff_hz(-1.0, 1e-6)

    with pytest.raises(ValueError, match="c_farads must be > 0"):
        compute_cutoff_hz(1000.0, 0.0)

    with pytest.raises(ValueError, match="c_farads must be > 0"):
        compute_cutoff_hz(1000.0, -1e-6)


def test_gain_lowpass_below_zero_db_at_fc():
    comps = {"R": 1000.0, "C": 1.59e-7}
    g = gain_db("low_pass", comps, 1000.0)
    assert g < 0


def test_compute_normalized_cost_minimum_is_zero():
    cost = compute_normalized_cost(100.0, 1e-10, 100.0, 1_000_000.0, 1e-10, 1e-3)
    assert 0.0 <= cost < 0.01


def test_compute_normalized_cost_maximum_is_one():
    cost = compute_normalized_cost(1_000_000.0, 1e-3, 100.0, 1_000_000.0, 1e-10, 1e-3)
    assert 0.99 < cost <= 1.0


def test_compute_normalized_cost_mixed_is_midrange():
    cost = compute_normalized_cost(100.0, 1e-3, 100.0, 1_000_000.0, 1e-10, 1e-3)
    assert 0.45 < cost < 0.55


def test_compute_normalized_cost_log_midpoint_is_half():
    cost = compute_normalized_cost(10_000.0, 3.1622776601683795e-7, 100.0, 1_000_000.0, 1e-10, 1e-3)
    assert round(cost, 6) == 0.5


def test_normalize_log_value_maps_exact_bounds():
    assert normalize_log_value(100.0, 100.0, 1_000_000.0) == 0.0
    assert normalize_log_value(1_000_000.0, 100.0, 1_000_000.0) == 1.0


@pytest.mark.parametrize(
    ("min_value", "max_value", "message"),
    [
        (0.0, 10.0, "min_value must be > 0"),
        (-1.0, 10.0, "min_value must be > 0"),
        (10.0, 10.0, "min_value must be < max_value"),
        (10.0, 1.0, "min_value must be < max_value"),
    ],
)
def test_normalize_log_value_rejects_invalid_ranges(min_value, max_value, message):
    with pytest.raises(ValueError, match=message):
        normalize_log_value(5.0, min_value, max_value)


def test_evaluate_circuit_state_computes_reward_and_done():
    evaluation = evaluate_circuit_state(
        1000.0,
        1.59e-7,
        1000.0,
        1,
        8,
        SUCCESS_TOLERANCE,
        100.0,
        1_000_000.0,
        1e-10,
        1e-3,
    )

    assert 0.0 <= evaluation["reward"] <= 1.0
    assert evaluation["done"] is True


def test_reward_is_explicitly_bounded_between_zero_and_one():
    very_good_reward = compute_reward(
        current_hz=1000.0,
        target_hz=1000.0,
        normalized_cost=0.0,
        step_count=0,
        max_steps=8,
    )
    very_bad_reward = compute_reward(
        current_hz=0.0,
        target_hz=1000.0,
        normalized_cost=1.0,
        step_count=8,
        max_steps=8,
    )

    assert very_good_reward <= 1.0
    assert very_bad_reward >= 0.0


def test_reward_improves_with_better_accuracy_when_cost_and_steps_match():
    closer_reward = compute_reward(
        current_hz=990.0,
        target_hz=1000.0,
        normalized_cost=0.25,
        step_count=2,
        max_steps=8,
    )
    farther_reward = compute_reward(
        current_hz=600.0,
        target_hz=1000.0,
        normalized_cost=0.25,
        step_count=2,
        max_steps=8,
    )

    assert closer_reward > farther_reward


def test_reward_improves_with_lower_cost_when_accuracy_and_steps_match():
    low_cost_reward = compute_reward(
        current_hz=980.0,
        target_hz=1000.0,
        normalized_cost=0.1,
        step_count=2,
        max_steps=8,
    )
    high_cost_reward = compute_reward(
        current_hz=980.0,
        target_hz=1000.0,
        normalized_cost=0.9,
        step_count=2,
        max_steps=8,
    )

    assert low_cost_reward > high_cost_reward


def test_reward_improves_with_fewer_steps_when_accuracy_and_cost_match():
    earlier_reward = compute_reward(
        current_hz=980.0,
        target_hz=1000.0,
        normalized_cost=0.25,
        step_count=1,
        max_steps=8,
    )
    later_reward = compute_reward(
        current_hz=980.0,
        target_hz=1000.0,
        normalized_cost=0.25,
        step_count=7,
        max_steps=8,
    )

    assert earlier_reward > later_reward


def test_error_reward_and_done_helpers_are_consistent():
    current_hz = 980.0
    normalized_error = compute_normalized_error(current_hz, 1000.0)
    reward = compute_reward(current_hz, 1000.0, 0.25, 2, 8)
    done = is_done(normalized_error, 2, 8, SUCCESS_TOLERANCE)

    assert normalized_error == 0.02
    assert 0.0 <= compute_step_efficiency(2, 8) <= 1.0
    assert 0.0 <= reward <= 1.0
    assert done is True


def test_compute_normalized_error_is_zero_for_perfect_match():
    assert compute_normalized_error(1000.0, 1000.0) == 0.0


def test_is_done_triggers_on_exact_tolerance_boundary():
    assert is_done(SUCCESS_TOLERANCE, 1, 8, SUCCESS_TOLERANCE) is True


def test_is_done_triggers_on_step_limit():
    assert is_done(0.5, 8, 8, SUCCESS_TOLERANCE) is True
