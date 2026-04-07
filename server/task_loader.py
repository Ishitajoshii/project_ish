"""Canonical task catalog loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from models import CircuitTaskSpec
from server.simulator import SUCCESS_TOLERANCE, WEIGHT_COST, WEIGHT_STEP

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
DEFAULT_TASK_IDS = [
    "lp_1khz_budget",
    "lp_10khz_budget",
    "hp_500hz_budget",
    "lp_2khz_low_cost",
]


def validate_task_spec(
    task: CircuitTaskSpec,
    source_path: Path | None = None,
    *,
    enforce_filename_match: bool = True,
) -> CircuitTaskSpec:
    """Enforce the frozen benchmark task contract."""

    label = str(source_path) if source_path is not None else task.task_id
    if not task.task_id.strip():
        raise ValueError(f"Task {label} must have a non-empty task_id")
    if task.target_hz <= 0:
        raise ValueError(f"Task {label} must have target_hz > 0")
    if task.max_steps <= 0:
        raise ValueError(f"Task {label} must have max_steps > 0")
    if task.success_tolerance_pct <= 0:
        raise ValueError(f"Task {label} must have success_tolerance_pct > 0")
    if task.success_tolerance_pct != SUCCESS_TOLERANCE * 100.0:
        raise ValueError(
            f"Task {label} must use success_tolerance_pct={SUCCESS_TOLERANCE * 100.0}"
        )
    if task.cost_weight != WEIGHT_COST:
        raise ValueError(f"Task {label} must use cost_weight={WEIGHT_COST}")
    if task.step_weight != WEIGHT_STEP:
        raise ValueError(f"Task {label} must use step_weight={WEIGHT_STEP}")
    if enforce_filename_match and source_path is not None and source_path.stem != task.task_id:
        raise ValueError(f"Task file {source_path.name} must match task_id={task.task_id}")
    return task


def load_task_file(path: Path) -> CircuitTaskSpec:
    """Read one JSON task file into a validated typed task spec."""

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    task = CircuitTaskSpec.model_validate(raw)
    return validate_task_spec(task, path)


def load_task(path: str | Path) -> CircuitTaskSpec:
    """Backward-compatible helper for loading one task by path."""

    return load_task_file(Path(path))


def load_tasks(tasks_dir: Path = TASKS_DIR) -> dict[str, CircuitTaskSpec]:
    """Load and validate every benchmark task JSON in a directory."""

    registry: dict[str, CircuitTaskSpec] = {}
    for path in sorted(tasks_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        task = validate_task_spec(
            CircuitTaskSpec.model_validate(raw),
            path,
            enforce_filename_match=False,
        )
        if task.task_id in registry:
            raise ValueError(f"Duplicate task_id detected: {task.task_id}")
        if path.stem != task.task_id:
            raise ValueError(f"Task file {path.name} must match task_id={task.task_id}")
        registry[task.task_id] = task
    return registry


def list_task_paths(task_dir: str | Path | None = None) -> list[Path]:
    """Return task file paths in deterministic catalog order."""

    base_dir = Path(task_dir) if task_dir is not None else TASKS_DIR
    tasks = load_tasks(base_dir)
    return [base_dir / f"{task_id}.json" for task_id in get_default_task_ids(tasks)]


def get_default_task_ids(tasks: dict[str, CircuitTaskSpec]) -> list[str]:
    """Return task ids in stable benchmark order."""

    ordered = [task_id for task_id in DEFAULT_TASK_IDS if task_id in tasks]
    remainder = sorted(task_id for task_id in tasks if task_id not in DEFAULT_TASK_IDS)
    return ordered + remainder


def list_task_ids(task_dir: str | Path | None = None) -> list[str]:
    """Return task identifiers in the same stable order used by inference."""

    base_dir = Path(task_dir) if task_dir is not None else TASKS_DIR
    return get_default_task_ids(load_tasks(base_dir))
