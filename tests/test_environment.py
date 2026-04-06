"""Quick confidence checks for reset/step flow and score availability."""

from inference import run_all_inference
from server.environment import CircuitEnvironment
from server.task_loader import list_task_ids, load_task


def test_reset_and_step_cycle():
    env = CircuitEnvironment(load_task("tasks/lp_1khz_budget.json"))
    obs = env.reset()
    assert obs.remaining_steps > 0

    obs2 = env.step({"action": "r_up"})
    assert obs2.remaining_steps == obs.remaining_steps - 1


def test_score_is_bounded():
    env = CircuitEnvironment(load_task("tasks/lp_1khz_budget.json"))
    env.reset()
    s = env.score()
    assert 0.0 <= s <= 1.0


def test_invalid_action_reports_last_action_error():
    env = CircuitEnvironment(load_task("tasks/lp_1khz_budget.json"))
    env.reset()

    obs = env.step({"action": "nope"})

    assert obs.last_action_error == "Unsupported action: nope"
    assert obs.remaining_steps == 8


def test_inference_enumerates_same_task_ids_as_loader():
    results = run_all_inference("tasks")
    assert [result["task_id"] for result in results] == list_task_ids("tasks")
