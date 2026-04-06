"""Quick confidence checks for simulator equations and action updates."""

from server.simulator import apply_action, cutoff_frequency_hz, gain_db


def test_apply_action_updates_component():
    comps = {"R": 1000.0, "C": 1e-7, "L": 1e-3}
    out = apply_action(comps, {"component": "R", "delta": 10.0})
    assert out["R"] == 1010.0


def test_cutoff_frequency_positive():
    fc = cutoff_frequency_hz({"R": 1000.0, "C": 1e-7, "L": 1e-3})
    assert fc > 0


def test_gain_lowpass_below_zero_db_at_fc():
    comps = {"R": 1000.0, "C": 1.59e-7, "L": 1e-3}
    g = gain_db("lowpass", comps, 1000.0)
    assert g < 0
