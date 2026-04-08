"""Confidence checks for deterministic baseline comparators."""

from models import CircuitObservation
from server.baselines import (
    BASELINE_ACTIONS,
    brute_force_baseline,
    choose_heuristic_action,
    heuristic_baseline,
    random_baseline,
    run_bruteforce_baseline,
    run_heuristic_baseline,
    run_random_baseline,
)
from server.environment import CircuitEnvironment
from server.task_loader import load_task


def test_random_baseline_is_deterministic_for_fixed_seed():
    task = load_task("tasks/lp_1khz_budget.json")

    first = run_random_baseline(CircuitEnvironment({task.task_id: task}), task.task_id, seed=7)
    second = run_random_baseline(CircuitEnvironment({task.task_id: task}), task.task_id, seed=7)

    assert first == second
    assert first["baseline_name"] == "random"
    assert first["task_id"] == task.task_id
    assert first["evaluations"] is None
    assert first["steps_used"] <= task.max_steps
    assert 0.0 <= first["score"] <= 1.0
    assert set(BASELINE_ACTIONS) == {"r_up", "r_down", "c_up", "c_down"}


def test_choose_heuristic_action_moves_cutoff_toward_target():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    obs = env.reset(task.task_id)

    assert obs.current_hz > obs.target_hz
    assert choose_heuristic_action(obs, 1, task) == "r_up"


def test_choose_heuristic_action_alternates_on_equal_cost_pressure():
    task = load_task("tasks/lp_1khz_budget.json")
    obs = CircuitObservation(
        task_id=task.task_id,
        circuit_type=task.circuit_type,
        target_hz=task.target_hz,
        current_r_ohms=10_000.0,
        current_c_farads=3.1622776601683795e-7,
        current_hz=1500.0,
        normalized_error=0.5,
        current_cost=0.5,
        remaining_steps=task.max_steps,
        last_action_error=None,
    )

    assert choose_heuristic_action(obs, 1, task) == "r_up"
    assert choose_heuristic_action(obs, 2, task) == "c_up"


def test_choose_heuristic_action_prefers_lower_cost_pressure_when_lowering_cutoff():
    task = load_task("tasks/lp_1khz_budget.json")
    tuned_obs = CircuitObservation(
        task_id=task.task_id,
        circuit_type=task.circuit_type,
        target_hz=task.target_hz,
        current_r_ohms=100_000.0,
        current_c_farads=1e-7,
        current_hz=1500.0,
        normalized_error=0.5,
        current_cost=0.5,
        remaining_steps=task.max_steps - 1,
        last_action_error=None,
    )

    assert tuned_obs.current_hz > tuned_obs.target_hz
    assert choose_heuristic_action(tuned_obs, 2, task) == "c_up"


def test_heuristic_baseline_returns_shared_payload_fields():
    task = load_task("tasks/lp_1khz_budget.json")

    result = run_heuristic_baseline(CircuitEnvironment({task.task_id: task}), task.task_id)

    assert result["baseline_name"] == "heuristic"
    assert result["task_id"] == task.task_id
    assert result["steps_used"] <= task.max_steps
    assert result["evaluations"] is None
    assert task.min_r_ohms <= result["current_r_ohms"] <= task.max_r_ohms
    assert task.min_c_farads <= result["current_c_farads"] <= task.max_c_farads
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["normalized_cost"] <= 1.0


def test_heuristic_baseline_completes_and_returns_compact_summary():
    task = load_task("tasks/lp_1khz_budget.json")

    result = run_heuristic_baseline(CircuitEnvironment({task.task_id: task}), task.task_id)

    assert result["task_id"] == task.task_id
    assert 0.0 <= result["score"] <= 1.0
    assert result["steps_used"] <= task.max_steps


def test_heuristic_chooser_always_returns_valid_action():
    task = load_task("tasks/lp_1khz_budget.json")
    observations = [
        CircuitObservation(
            task_id=task.task_id,
            circuit_type=task.circuit_type,
            target_hz=task.target_hz,
            current_r_ohms=1_000.0,
            current_c_farads=1e-7,
            current_hz=1_500.0,
            normalized_error=0.5,
            current_cost=0.3,
            remaining_steps=task.max_steps,
            last_action_error=None,
        ),
        CircuitObservation(
            task_id=task.task_id,
            circuit_type=task.circuit_type,
            target_hz=task.target_hz,
            current_r_ohms=10_000.0,
            current_c_farads=1e-6,
            current_hz=500.0,
            normalized_error=0.5,
            current_cost=0.7,
            remaining_steps=task.max_steps - 1,
            last_action_error=None,
        ),
    ]

    for step_count, observation in enumerate(observations, start=1):
        assert choose_heuristic_action(observation, step_count, task) in BASELINE_ACTIONS


def test_bruteforce_baseline_reports_grid_evaluations_and_best_candidate():
    task = load_task("tasks/lp_1khz_budget.json")

    result = run_bruteforce_baseline(task, num_r_points=4, num_c_points=5)

    assert result["baseline_name"] == "bruteforce"
    assert result["task_id"] == task.task_id
    assert result["evaluations"] == 20
    assert result["steps_used"] == 0
    assert task.min_r_ohms <= result["current_r_ohms"] <= task.max_r_ohms
    assert task.min_c_farads <= result["current_c_farads"] <= task.max_c_farads
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["normalized_error"]


def test_legacy_wrapper_functions_match_structured_scores():
    task = load_task("tasks/lp_1khz_budget.json")

    random_result = run_random_baseline(CircuitEnvironment({task.task_id: task}), task.task_id, seed=3)
    heuristic_result = run_heuristic_baseline(CircuitEnvironment({task.task_id: task}), task.task_id)
    brute_force_result = run_bruteforce_baseline(task)

    assert random_baseline(task, seed=3) == random_result["score"]
    assert heuristic_baseline(task) == heuristic_result["score"]
    assert brute_force_baseline(task) == brute_force_result["score"]
