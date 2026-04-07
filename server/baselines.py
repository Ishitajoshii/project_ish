"""Baseline comparators: random, heuristic, and tiny brute-force sweeps."""

from __future__ import annotations

import random

from models import CircuitAction
from models import CircuitTaskSpec
from server.environment import CircuitEnvironment
from server.simulator import valid_actions


def random_baseline(task: CircuitTaskSpec, seed: int = 0) -> float:
    """Run random actions for one episode and return final score."""

    random.seed(seed)
    env = CircuitEnvironment({task.task_id: task})
    env.reset(task.task_id)
    for _ in range(task.max_steps):
        if env.is_done:
            break
        env.step(CircuitAction(action=random.choice(valid_actions())))
    return env.score()


def heuristic_baseline(task: CircuitTaskSpec) -> float:
    """Use frequency-direction heuristic to iteratively tune resistor value."""

    env = CircuitEnvironment({task.task_id: task})
    obs = env.reset(task.task_id)
    while not env.is_done:
        action = "r_up" if obs.current_hz > task.target_hz else "r_down"
        obs, _, _ = env.step(CircuitAction(action=action))
    return env.score()


def brute_force_baseline(task: CircuitTaskSpec) -> float:
    """Try a tiny grid over R deltas and keep the best resulting score."""

    best = 0.0
    for action in ("r_up", "r_down", "c_up", "c_down"):
        env = CircuitEnvironment({task.task_id: task})
        env.reset(task.task_id)
        for _ in range(task.max_steps):
            if env.is_done:
                break
            env.step(CircuitAction(action=action))
        best = max(best, env.score())
    return best
