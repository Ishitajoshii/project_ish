"""Quick confidence checks for reset/step flow and score availability."""

from server.environment import CircuitEnvironment
from server.task_loader import load_task


def test_reset_and_step_cycle():
    env = CircuitEnvironment(load_task("tasks/lp_1khz_budget.json"))
    obs = env.reset()
    assert obs.remaining_steps > 0

    obs2 = env.step({"component": "R", "delta": 0.2})
    assert obs2.remaining_steps == obs.remaining_steps - 1


def test_score_is_bounded():
    env = CircuitEnvironment(load_task("tasks/lp_1khz_budget.json"))
    env.reset()
    s = env.score()
    assert 0.0 <= s <= 1.0
