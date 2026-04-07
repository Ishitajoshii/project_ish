"""Quick confidence checks for reset/step flow and score availability."""

import pytest

from models import CircuitAction
from inference import run_all_inference
from server.environment import CircuitEnvironment
from server.grader import normalized_score
from server.task_loader import list_task_ids, load_task


def test_reset_and_step_cycle():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})
    obs = env.reset(task["task_id"])
    assert obs.remaining_steps > 0

    obs2 = env.step(CircuitAction(action="r_up"))
    assert obs2.remaining_steps == obs.remaining_steps - 1


def test_score_is_bounded():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})
    env.reset(task["task_id"])
    s = env.score()
    assert 0.0 <= s <= 1.0


def test_invalid_action_reports_last_action_error():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})
    obs = env.reset(task["task_id"])

    obs = env.step({"action": "nope"})

    assert obs.last_action_error == "invalid action: nope"
    assert obs.remaining_steps == 7
    assert env.score() > 0.0


def test_inference_enumerates_same_task_ids_as_loader():
    results = run_all_inference("tasks")
    assert [result["task_id"] for result in results] == list_task_ids("tasks")


def test_final_score_uses_best_reward_seen():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})
    env.reset(task["task_id"])

    first = env.step(CircuitAction(action="r_up"))
    best_after_first = normalized_score(
        first.normalized_error,
        first.current_cost,
        1,
        8,
    )
    assert env.score() == best_after_first

    env.step(CircuitAction(action="r_down"))
    assert env.score() == best_after_first


def test_low_cost_task_starts_expensive_and_can_improve_by_moving_r_down():
    task = load_task("tasks/lp_2khz_low_cost.json")
    env = CircuitEnvironment({task["task_id"]: task})
    start = env.reset(task["task_id"])

    assert start.current_hz < start.target_hz
    initial_cost = start.current_cost

    latest = start
    for _ in range(6):
        latest = env.step(CircuitAction(action="r_down"))

    assert latest.current_cost < initial_cost
    assert latest.normalized_error <= 0.02


def test_state_returns_typed_episode_summary():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})
    env.reset(task["task_id"])
    env.step(CircuitAction(action="r_up"))

    state = env.state()
    assert state.task_id == task["task_id"]
    assert state.step_count == 1
    assert state.best_score == env.score()
    assert state.done is False


def test_step_before_reset_raises_clear_error():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})

    with pytest.raises(RuntimeError, match="environment must be reset before stepping"):
        env.step(CircuitAction(action="r_up"))


def test_step_after_done_raises_clear_error():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task["task_id"]: task})
    env.reset(task["task_id"])

    while not env.is_done:
        env.step(CircuitAction(action="r_up"))

    with pytest.raises(RuntimeError, match="episode is already done; call reset\\(\\) to start a new task"):
        env.step(CircuitAction(action="r_up"))
