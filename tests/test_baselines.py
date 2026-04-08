"""Confidence checks for deterministic baseline comparators."""

from server.baselines import (
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
    assert first["evaluations"] == first["steps_used"]
    assert 0.0 <= first["score"] <= 1.0


def test_choose_heuristic_action_moves_cutoff_toward_target():
    task = load_task("tasks/lp_1khz_budget.json")
    env = CircuitEnvironment({task.task_id: task})
    obs = env.reset(task.task_id)

    assert obs.current_hz > obs.target_hz
    assert choose_heuristic_action(obs) == "r_up"


def test_heuristic_baseline_returns_shared_payload_fields():
    task = load_task("tasks/lp_1khz_budget.json")

    result = run_heuristic_baseline(CircuitEnvironment({task.task_id: task}), task.task_id)

    assert result["baseline_name"] == "heuristic"
    assert result["task_id"] == task.task_id
    assert result["steps_used"] <= task.max_steps
    assert result["evaluations"] == result["steps_used"]
    assert task.min_r_ohms <= result["best_r_ohms"] <= task.max_r_ohms
    assert task.min_c_farads <= result["best_c_farads"] <= task.max_c_farads
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["normalized_cost"] <= 1.0


def test_bruteforce_baseline_reports_grid_evaluations_and_best_candidate():
    task = load_task("tasks/lp_1khz_budget.json")

    result = run_bruteforce_baseline(task, num_r_points=4, num_c_points=5)

    assert result["baseline_name"] == "bruteforce"
    assert result["task_id"] == task.task_id
    assert result["evaluations"] == 20
    assert result["steps_used"] == task.max_steps
    assert task.min_r_ohms <= result["best_r_ohms"] <= task.max_r_ohms
    assert task.min_c_farads <= result["best_c_farads"] <= task.max_c_farads
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
