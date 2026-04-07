"""Evaluator-facing script that emits compact JSON lines to stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models import CircuitAction
from server.environment import CircuitEnvironment
from server.task_loader import get_task_ids_in_order, load_task_file, load_tasks


def run_inference(task_file: str) -> dict:
    """Run a deterministic heuristic and return evaluator payload."""

    task = load_task_file(Path(task_file))
    env = CircuitEnvironment({task.task_id: task})
    obs = env.reset(task.task_id)

    # Minimal deterministic policy: adjust R in the direction that moves cutoff
    # frequency toward the target.
    while not env.is_done:
        action = "r_up" if obs.current_hz > task.target_hz else "r_down"
        obs, _, _ = env.step(CircuitAction(action=action))

    score = env.score()
    return {
        "task_id": task.task_id,
        "score": score,
        "details": {
            "final_output_hz": obs.current_hz,
            "normalized_error": obs.normalized_error,
            "cost": obs.current_cost,
            "solved": env.is_done and obs.normalized_error <= 0.02,
        },
    }


def run_all_inference(task_dir: str | None = None) -> list[dict]:
    """Run inference across the deterministic benchmark task set."""

    base_dir = Path(task_dir) if task_dir is not None else Path("tasks")
    tasks = load_tasks(base_dir)
    ordered_ids = get_task_ids_in_order(tasks)
    return [run_inference(str(base_dir / f"{task_id}.json")) for task_id in ordered_ids]


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
