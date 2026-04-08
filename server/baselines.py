"""Deterministic baseline comparators for RC tuning benchmark tasks."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

from models import CircuitAction, CircuitObservation, CircuitTaskSpec
from server.environment import CircuitEnvironment
from server.grader import is_success
from server.simulator import SUCCESS_TOLERANCE, evaluate_circuit_state, normalize_log_value

BASELINE_ACTIONS = ["r_up", "r_down", "c_up", "c_down"]


def _log_space(min_value: float, max_value: float, num_points: int) -> list[float]:
    """Return a deterministic base-10 log grid including both endpoints."""

    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    if num_points == 1:
        return [min_value]

    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    step = (log_max - log_min) / (num_points - 1)
    return [10 ** (log_min + (index * step)) for index in range(num_points)]


def build_baseline_result(
    *,
    baseline_name: str,
    task_id: str,
    score: float,
    success: bool,
    steps_used: int,
    evaluations: int | None,
    achieved_hz: float,
    best_r_ohms: float,
    best_c_farads: float,
    normalized_error: float,
    normalized_cost: float,
) -> dict[str, Any]:
    """Build one shared baseline payload shape."""

    return {
        "baseline_name": baseline_name,
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps_used": steps_used,
        "evaluations": evaluations,
        "achieved_hz": achieved_hz,
        "best_r_ohms": best_r_ohms,
        "best_c_farads": best_c_farads,
        "normalized_error": normalized_error,
        "normalized_cost": normalized_cost,
    }


def run_baseline_episode(
    env: CircuitEnvironment,
    task_id: str,
    choose_action: Callable[[CircuitObservation, int, CircuitTaskSpec], str],
    *,
    baseline_name: str,
) -> dict[str, Any]:
    """Run one deterministic policy against the environment until termination."""

    observation = env.reset(task_id)
    task = env._require_task()
    while not env.is_done:
        action_name = choose_action(observation, env.step_count + 1, task)
        observation, _, done = env.step(CircuitAction(action=action_name))
        if done:
            break

    final_state = env.state()
    score = env.score()
    return build_baseline_result(
        baseline_name=baseline_name,
        task_id=task_id,
        score=score,
        success=is_success(score),
        steps_used=final_state.step_count,
        evaluations=None,
        achieved_hz=float(final_state.best_hz),
        best_r_ohms=float(final_state.best_r_ohms),
        best_c_farads=float(final_state.best_c_farads),
        normalized_error=float(final_state.best_normalized_error),
        normalized_cost=float(final_state.best_normalized_cost),
    )


def run_random_baseline(
    env: CircuitEnvironment,
    task_id: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a seeded uniform-random policy over the discrete action space."""

    rng = random.Random(seed)
    return run_baseline_episode(
        env,
        task_id,
        lambda _observation, _step_count, _task: rng.choice(BASELINE_ACTIONS),
        baseline_name="random",
    )


def _prefer_r_on_step(step_count: int) -> bool:
    """Use step parity as a deterministic fallback tie-breaker."""

    return step_count % 2 == 1


def choose_heuristic_action(
    observation: CircuitObservation,
    step_count: int,
    task: CircuitTaskSpec,
) -> str:
    """Choose a deterministic action using frequency direction and cost pressure."""

    r_pressure = normalize_log_value(
        observation.current_r_ohms,
        task.min_r_ohms,
        task.max_r_ohms,
    )
    c_pressure = normalize_log_value(
        observation.current_c_farads,
        task.min_c_farads,
        task.max_c_farads,
    )

    if observation.current_hz > observation.target_hz:
        if abs(r_pressure - c_pressure) <= 1e-12:
            return "r_up" if _prefer_r_on_step(step_count) else "c_up"
        return "r_up" if r_pressure < c_pressure else "c_up"

    if abs(r_pressure - c_pressure) <= 1e-12:
        return "r_down" if _prefer_r_on_step(step_count) else "c_down"
    return "r_down" if r_pressure > c_pressure else "c_down"


def run_heuristic_baseline(
    env: CircuitEnvironment,
    task_id: str,
) -> dict[str, Any]:
    """Run a simple frequency-direction heuristic through the real environment."""

    return run_baseline_episode(
        env,
        task_id,
        choose_heuristic_action,
        baseline_name="heuristic",
    )


def run_bruteforce_baseline(
    task: CircuitTaskSpec,
    num_r_points: int = 10,
    num_c_points: int = 10,
) -> dict[str, Any]:
    """Evaluate a coarse log-space RC grid directly with simulator helpers."""

    r_values = _log_space(task.min_r_ohms, task.max_r_ohms, num_r_points)
    c_values = _log_space(task.min_c_farads, task.max_c_farads, num_c_points)

    best_result: dict[str, float | bool] | None = None
    best_r = task.initial_r_ohms
    best_c = task.initial_c_farads
    evaluations = 0

    for r_ohms in r_values:
        for c_farads in c_values:
            evaluations += 1
            candidate = evaluate_circuit_state(
                r_ohms=r_ohms,
                c_farads=c_farads,
                target_hz=task.target_hz,
                step_count=task.max_steps,
                max_steps=task.max_steps,
                success_tolerance=SUCCESS_TOLERANCE,
                min_r_ohms=task.min_r_ohms,
                max_r_ohms=task.max_r_ohms,
                min_c_farads=task.min_c_farads,
                max_c_farads=task.max_c_farads,
            )
            if best_result is None or float(candidate["reward"]) > float(best_result["reward"]):
                best_result = candidate
                best_r = r_ohms
                best_c = c_farads

    assert best_result is not None
    score = float(best_result["reward"])
    return build_baseline_result(
        baseline_name="bruteforce",
        task_id=task.task_id,
        score=score,
        success=is_success(score),
        steps_used=0,
        evaluations=evaluations,
        achieved_hz=float(best_result["current_hz"]),
        best_r_ohms=best_r,
        best_c_farads=best_c,
        normalized_error=float(best_result["normalized_error"]),
        normalized_cost=float(best_result["normalized_cost"]),
    )


def random_baseline(task: CircuitTaskSpec, seed: int = 0) -> float:
    """Backward-compatible wrapper returning only the baseline score."""

    env = CircuitEnvironment({task.task_id: task})
    return float(run_random_baseline(env, task.task_id, seed=seed)["score"])


def heuristic_baseline(task: CircuitTaskSpec) -> float:
    """Backward-compatible wrapper returning only the baseline score."""

    env = CircuitEnvironment({task.task_id: task})
    return float(run_heuristic_baseline(env, task.task_id)["score"])


def brute_force_baseline(task: CircuitTaskSpec) -> float:
    """Backward-compatible wrapper returning only the brute-force score."""

    return float(run_bruteforce_baseline(task)["score"])
