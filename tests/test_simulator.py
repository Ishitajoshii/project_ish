"""Quick confidence checks for simulator equations and action updates."""

from server.simulator import apply_action, component_cost, cutoff_frequency_hz, gain_db


def test_apply_action_updates_component_multiplicatively():
    comps = {"R": 1000.0, "C": 1e-7}
    bounds = {"R": (100.0, 1_000_000.0), "C": (1e-10, 1e-3)}
    out = apply_action(comps, "r_up", bounds)
    assert out["R"] == 1200.0


def test_cutoff_frequency_positive():
    fc = cutoff_frequency_hz({"R": 1000.0, "C": 1e-7})
    assert fc > 0


def test_gain_lowpass_below_zero_db_at_fc():
    comps = {"R": 1000.0, "C": 1.59e-7}
    g = gain_db("low_pass", comps, 1000.0)
    assert g < 0


def test_component_cost_is_log_normalized():
    bounds = {"R": (100.0, 1_000_000.0), "C": (1e-10, 1e-3)}
    cheapest = component_cost({"R": 100.0, "C": 1e-10}, bounds)
    middle = component_cost({"R": 10_000.0, "C": 3.1622776601683795e-7}, bounds)
    priciest = component_cost({"R": 1_000_000.0, "C": 1e-3}, bounds)

    assert cheapest == 0.0
    assert round(middle, 6) == 0.5
    assert priciest == 1.0
