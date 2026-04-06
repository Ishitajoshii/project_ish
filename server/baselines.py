"""Baseline comparators: random, heuristic, and tiny brute-force sweeps."""

from __future__ import annotations

import random
from typing import Any

from server.environment import CircuitEnvironment


def random_baseline(task: dict[str, Any], seed: int = 0) -> float:
    """Run random actions for one episode and return final score."""

    random.seed(seed)
    env = CircuitEnvironment(task)
    env.reset()
    for _ in range(task["max_steps"]):
        if env.is_done:
            break
        action = {
            "component": random.choice(["R", "C"]),
            "delta": random.choice([-0.2, -0.1, 0.1, 0.2]),
        }
        env.step(action)
    return env.score()


def heuristic_baseline(task: dict[str, Any]) -> float:
    """Use frequency-direction heuristic to iteratively tune resistor value."""

    env = CircuitEnvironment(task)
    obs = env.reset()
    while not env.is_done:
        delta = 0.2 if obs.current_output_hz > task["target_hz"] else -0.2
        obs = env.step({"component": "R", "delta": delta})
    return env.score()


def brute_force_baseline(task: dict[str, Any]) -> float:
    """Try a tiny grid over R deltas and keep the best resulting score."""

    best = 0.0
    for delta in (-0.2, -0.1, 0.1, 0.2):
        env = CircuitEnvironment(task)
        env.reset()
        for _ in range(task["max_steps"]):
            if env.is_done:
                break
            env.step({"component": "R", "delta": delta})
        best = max(best, env.score())
    return best
