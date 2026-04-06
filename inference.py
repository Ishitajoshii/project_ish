"""Evaluator-facing script that emits exactly one JSON line to stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from server.environment import CircuitEnvironment
from server.task_loader import load_task


def run_inference(task_file: str) -> dict:
    """Run a deterministic heuristic and return evaluator payload."""

    task = load_task(task_file)
    env = CircuitEnvironment(task)
    obs = env.reset()

    # Minimal deterministic policy: adjust R in the direction that moves cutoff
    # frequency toward the target.
    while not env.is_done:
        delta = 0.2 if obs.current_output_hz > task["target_hz"] else -0.2
        obs = env.step({"component": "R", "delta": delta})

    score = env.score()
    return {
        "task_id": task["task_id"],
        "score": score,
        "details": {
            "final_output_hz": obs.current_output_hz,
            "normalized_error": obs.normalized_error,
            "cost": obs.current_cost,
            "solved": obs.solved,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Path to task JSON")
    args = parser.parse_args()

    result = run_inference(args.task)

    # Strict stdout contract: emit exactly one compact JSON object line.
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
