"""Quick confidence checks for reset/step flow and score availability."""

import json

import pytest

from models import CircuitAction
from inference import run_all_inference
from server.environment import CircuitEnvironment
from server.simulator import compute_reward
from server.task_loader import list_task_ids, load_task


def proposal_json(action: str) -> str:
    return json.dumps(
        {
            "action": action,
            "objective": "Tighten the target mismatch",
            "rationale": "Use the strongest evaluated move on the current board.",
            "expected_outcome": "Improve the engineering tradeoff.",
            "confidence": 0.8,
        }
    )


class FakeResponses:
    def __init__(self, output_text: str = proposal_json("c_up")) -> None:
        self.output_text = output_text

    def create(self, **kwargs):
        return type("Response", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, output_text: str = proposal_json("c_up")) -> None:
        self.responses = FakeResponses(output_text=output_text)


def test_reset_and_step_cycle():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    obs = env.reset(task.task_id)
    assert obs.remaining_steps > 0

    obs2, reward, done = env.step(CircuitAction(action="r_up"))
    assert obs2.remaining_steps == obs.remaining_steps - 1
    assert 0.0 <= reward <= 1.0
    assert done is False


def test_score_is_bounded():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    env.reset(task.task_id)
    s = env.score()
    assert 0.0 <= s <= 1.0


def test_reset_initializes_clean_state():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})

    obs = env.reset(task.task_id)

    assert env.step_count == 0
    assert env.cumulative_reward == 0.0
    assert env.best_score == 0.0
    assert env.best_r_ohms == obs.current_r_ohms
    assert env.best_c_farads == obs.current_c_farads
    assert env.best_hz == obs.current_hz
    assert env.best_normalized_error == obs.normalized_error
    assert env.best_normalized_cost == obs.current_cost
    assert env.done is False


def test_invalid_action_reports_last_action_error():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    start = env.reset(task.task_id)

    obs, reward, done = env.step({"action": "nope"})

    assert obs.last_action_error == "invalid action: nope"
    assert obs.current_r_ohms == start.current_r_ohms
    assert obs.current_c_farads == start.current_c_farads
    assert obs.remaining_steps == 7
    assert 0.0 <= reward <= 1.0
    assert done is False
    assert env.score() > 0.0


def test_step_increments_step_count_before_reward_logic():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    env.reset(task.task_id)

    obs, reward, _ = env.step(CircuitAction(action="r_up"))

    expected_reward = compute_reward(
        current_hz=obs.current_hz,
        target_hz=task.target_hz,
        normalized_cost=obs.current_cost,
        step_count=1,
        max_steps=task.max_steps,
    )

    assert env.step_count == 1
    assert reward == expected_reward


def test_inference_enumerates_same_task_ids_as_loader(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    results = run_all_inference("tasks", client=FakeClient(output_text=proposal_json("c_up")))
    assert [result["task_id"] for result in results] == list_task_ids("tasks")


def test_final_score_uses_best_reward_seen():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    env.reset(task.task_id)

    _, first_reward, first_done = env.step(CircuitAction(action="r_up"))
    _, second_reward, _ = env.step(CircuitAction(action="r_down"))

    assert first_done is False
    assert first_reward > second_reward
    assert env.score() == first_reward
    assert env.best_r_ohms == task.initial_r_ohms * env.action_scale_factor
    assert env.best_c_farads == task.initial_c_farads


def test_state_exposes_best_state_snapshot():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    initial = env.reset(task.task_id)

    env.step(CircuitAction(action="r_up"))
    state = env.state()

    assert state.best_r_ohms == initial.current_r_ohms * env.action_scale_factor
    assert state.best_c_farads == initial.current_c_farads
    assert state.best_hz is not None
    assert state.best_normalized_error is not None
    assert state.best_normalized_cost is not None
    assert state.target_hz == task.target_hz
    assert state.circuit_type == task.circuit_type
    assert state.current_normalized_error == env.normalized_error
    assert state.current_cost == env.current_cost


def test_low_cost_task_starts_expensive_and_can_improve_by_moving_r_down():
    task = load_task("tasks/lp_2khz_low_cost.json")
    env = CircuitEnvironment({task.task_id: task})
    start = env.reset(task.task_id)

    assert start.current_hz < start.target_hz
    initial_cost = start.current_cost

    latest = start
    for _ in range(6):
        latest, _, _ = env.step(CircuitAction(action="r_down"))

    assert latest.current_cost < initial_cost
    assert latest.normalized_error <= 0.02


def test_state_returns_typed_episode_summary():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    env.reset(task.task_id)
    env.step(CircuitAction(action="r_up"))

    state = env.state()
    assert state.task_id == task.task_id
    assert state.circuit_type == task.circuit_type
    assert state.target_hz == task.target_hz
    assert state.step_count == 1
    assert state.best_score == env.score()
    assert state.done is False
    assert state.current_hz == env.current_hz
    assert state.current_normalized_error == env.normalized_error
    assert state.current_cost == env.current_cost
    assert state.best_r_ohms is not None
    assert state.best_c_farads is not None


def test_step_before_reset_raises_clear_error():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})

    with pytest.raises(RuntimeError, match="environment must be reset before use"):
        env.step(CircuitAction(action="r_up"))


def test_step_after_done_raises_clear_error():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    env.reset(task.task_id)

    while not env.is_done:
        env.step(CircuitAction(action="r_up"))

    with pytest.raises(RuntimeError, match="episode is already done; call reset\\(\\) to start a new task"):
        env.step(CircuitAction(action="r_up"))
