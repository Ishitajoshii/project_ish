"""Task loader for deterministic benchmark JSON definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from server.simulator import SUCCESS_TOLERANCE, WEIGHT_COST, WEIGHT_STEP

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"

REQUIRED_FIELDS = {
    "task_id",
    "circuit_type",
    "target_hz",
    "initial_r_ohms",
    "initial_c_farads",
    "min_r_ohms",
    "max_r_ohms",
    "min_c_farads",
    "max_c_farads",
    "max_steps",
    "success_tolerance_pct",
    "cost_weight",
    "step_weight",
}


def _validate_task(task: dict[str, Any], task_path: Path) -> None:
    """Raise a clear error if a task definition does not match the RC schema."""

    missing = sorted(REQUIRED_FIELDS - task.keys())
    if missing:
        raise ValueError(f"Task {task_path} is missing required fields: {', '.join(missing)}")

    if task["circuit_type"] not in {"low_pass", "high_pass"}:
        raise ValueError(f"Task {task_path} has unsupported circuit_type={task['circuit_type']!r}")

    if float(task["min_r_ohms"]) >= float(task["max_r_ohms"]):
        raise ValueError(f"Task {task_path} has invalid resistor bounds")
    if float(task["min_c_farads"]) >= float(task["max_c_farads"]):
        raise ValueError(f"Task {task_path} has invalid capacitor bounds")

    if not (float(task["min_r_ohms"]) <= float(task["initial_r_ohms"]) <= float(task["max_r_ohms"])):
        raise ValueError(f"Task {task_path} has initial_r_ohms outside bounds")
    if not (float(task["min_c_farads"]) <= float(task["initial_c_farads"]) <= float(task["max_c_farads"])):
        raise ValueError(f"Task {task_path} has initial_c_farads outside bounds")

    if int(task["max_steps"]) <= 0:
        raise ValueError(f"Task {task_path} must have max_steps > 0")
    if float(task["success_tolerance_pct"]) < 0.0:
        raise ValueError(f"Task {task_path} must have non-negative success_tolerance_pct")

    total_weight = float(task["cost_weight"]) + float(task["step_weight"])
    if total_weight > 1.0:
        raise ValueError(f"Task {task_path} has cost_weight + step_weight > 1.0")
    if float(task["success_tolerance_pct"]) != SUCCESS_TOLERANCE * 100.0:
        raise ValueError(
            f"Task {task_path} must use success_tolerance_pct={SUCCESS_TOLERANCE * 100.0}"
        )
    if float(task["cost_weight"]) != WEIGHT_COST:
        raise ValueError(f"Task {task_path} must use cost_weight={WEIGHT_COST}")
    if float(task["step_weight"]) != WEIGHT_STEP:
        raise ValueError(f"Task {task_path} must use step_weight={WEIGHT_STEP}")


def load_task(path: str | Path) -> dict[str, Any]:
    """Load and return one task definition as a dictionary."""

    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    _validate_task(task, task_path)
    return task


def list_task_paths(task_dir: str | Path | None = None) -> list[Path]:
    """Return the deterministic list of benchmark task files."""

    base_dir = Path(task_dir) if task_dir is not None else TASKS_DIR
    return sorted(base_dir.glob("*.json"))


def list_task_ids(task_dir: str | Path | None = None) -> list[str]:
    """Return the task identifiers in the same order used by inference."""

    return [load_task(path)["task_id"] for path in list_task_paths(task_dir)]
