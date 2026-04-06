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
        action = {
            "component": random.choice(["R", "C", "L"]),
            "delta": random.uniform(-0.1, 0.1),
        }
        env.step(action)
    return env.score()


def heuristic_baseline(task: dict[str, Any]) -> float:
    """Use sign-of-gain heuristic to iteratively tune resistor value."""

    env = CircuitEnvironment(task)
    obs = env.reset()
    while not env.is_done:
        delta = -0.05 if obs.gain_db > -3.0 else 0.05
        obs = env.step({"component": "R", "delta": delta})
    return env.score()


def brute_force_baseline(task: dict[str, Any]) -> float:
    """Try a tiny grid over R deltas and keep the best resulting score."""

    best = 0.0
    for delta in (-0.1, -0.05, 0.0, 0.05, 0.1):
        env = CircuitEnvironment(task)
        env.reset()
        for _ in range(task["max_steps"]):
            env.step({"component": "R", "delta": delta})
        best = max(best, env.score())
    return best
