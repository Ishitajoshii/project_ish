"""Baseline comparators: random, heuristic, and tiny brute-force sweeps."""

from __future__ import annotations

import random
from typing import Any

from server.environment import CircuitEnvironment
from server.simulator import valid_actions


def random_baseline(task: dict[str, Any], seed: int = 0) -> float:
    """Run random actions for one episode and return final score."""

    random.seed(seed)
    env = CircuitEnvironment(task)
    env.reset()
    for _ in range(task["max_steps"]):
        if env.is_done:
            break
        env.step({"action": random.choice(valid_actions())})
    return env.score()


def heuristic_baseline(task: dict[str, Any]) -> float:
    """Use frequency-direction heuristic to iteratively tune resistor value."""

    env = CircuitEnvironment(task)
    obs = env.reset()
    while not env.is_done:
        action = "r_up" if obs.current_output_hz > task["target_hz"] else "r_down"
        obs = env.step({"action": action})
    return env.score()


def brute_force_baseline(task: dict[str, Any]) -> float:
    """Try a tiny grid over R deltas and keep the best resulting score."""

    best = 0.0
    for action in ("r_up", "r_down", "c_up", "c_down"):
        env = CircuitEnvironment(task)
        env.reset()
        for _ in range(task["max_steps"]):
            if env.is_done:
                break
            env.step({"action": action})
        best = max(best, env.score())
    return best
