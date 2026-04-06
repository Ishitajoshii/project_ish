"""Evaluator-facing script that emits compact JSON lines to stdout."""

from __future__ import annotations

import argparse
import json

from server.environment import CircuitEnvironment
from server.task_loader import list_task_paths, load_task


def run_inference(task_file: str) -> dict:
    """Run a deterministic heuristic and return evaluator payload."""

    task = load_task(task_file)
    env = CircuitEnvironment(task)
    obs = env.reset()

    # Minimal deterministic policy: adjust R in the direction that moves cutoff
    # frequency toward the target.
    while not env.is_done:
        action = "r_up" if obs.current_output_hz > task["target_hz"] else "r_down"
        obs = env.step({"action": action})

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


def run_all_inference(task_dir: str | None = None) -> list[dict]:
    """Run inference across the deterministic benchmark task set."""

    return [run_inference(str(task_path)) for task_path in list_task_paths(task_dir)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Path to one task JSON; omit to run all benchmark tasks")
    args = parser.parse_args()

    results = [run_inference(args.task)] if args.task else run_all_inference()

    # Emit one compact JSON object line per task for evaluator-friendly logs.
    for result in results:
        print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
