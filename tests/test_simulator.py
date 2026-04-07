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


def test_apply_action_clamps_to_bounds():
    new_r, new_c, error = apply_action(
        1_000_000.0,
        1e-9,
        "r_up",
        ACTION_SCALE_FACTOR,
        100.0,
        1_000_000.0,
        1e-9,
        1e-3,
    )
    assert error is None
    assert new_r == 1_000_000.0
    assert new_c == 1e-9


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

    with pytest.raises(ValueError, match="c_farads must be > 0"):
        compute_cutoff_hz(1000.0, 0.0)


def test_gain_lowpass_below_zero_db_at_fc():
    comps = {"R": 1000.0, "C": 1.59e-7}
    g = gain_db("low_pass", comps, 1000.0)
    assert g < 0


def test_component_cost_is_log_normalized():
    cheapest = compute_normalized_cost(100.0, 1e-10, 100.0, 1_000_000.0, 1e-10, 1e-3)
    middle = compute_normalized_cost(10_000.0, 3.1622776601683795e-7, 100.0, 1_000_000.0, 1e-10, 1e-3)
    priciest = compute_normalized_cost(1_000_000.0, 1e-3, 100.0, 1_000_000.0, 1e-10, 1e-3)

    assert cheapest == 0.0
    assert round(middle, 6) == 0.5
    assert priciest == 1.0


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


def test_error_reward_and_done_helpers_are_consistent():
    current_hz = 980.0
    normalized_error = compute_normalized_error(current_hz, 1000.0)
    reward = compute_reward(current_hz, 1000.0, 0.25, 2, 8)
    done = is_done(normalized_error, 2, 8, SUCCESS_TOLERANCE)

    assert normalized_error == 0.02
    assert 0.0 <= compute_step_efficiency(2, 8) <= 1.0
    assert 0.0 <= reward <= 1.0
    assert done is True
