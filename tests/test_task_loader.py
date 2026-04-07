"""Confidence checks for task catalog loading and validation."""

from pathlib import Path
import uuid

import pytest

from server.task_loader import DEFAULT_TASK_ORDER, get_task_ids_in_order, load_task, load_tasks


def test_load_task_returns_typed_spec():
    task = load_task("tasks/lp_1khz_budget.json")
    assert task.task_id == "lp_1khz_budget"
    assert task.target_hz == 1000.0


def test_load_tasks_returns_registry_keyed_by_task_id():
    tasks = load_tasks(Path("tasks"))
    assert set(DEFAULT_TASK_ORDER).issubset(tasks.keys())
    assert tasks["hp_500hz_budget"].circuit_type == "high_pass"


def test_get_task_ids_in_order_returns_stable_order():
    tasks = load_tasks(Path("tasks"))
    assert get_task_ids_in_order(tasks) == DEFAULT_TASK_ORDER


def test_load_tasks_rejects_duplicate_task_ids():
    src = Path("tasks")
    parent_dir = Path("tests") / f"_task_loader_tmp_{uuid.uuid4().hex}"
    sandbox_dir = parent_dir / "nested"
    sandbox_dir.mkdir(parents=True)
    try:
        for name in ("lp_1khz_budget.json", "lp_10khz_budget.json"):
            (sandbox_dir / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")

        duplicate = (src / "lp_1khz_budget.json").read_text(encoding="utf-8")
        (parent_dir / "lp_1khz_budget.json").write_text(duplicate, encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate task_id detected"):
            load_tasks(parent_dir)
    finally:
        for path in sandbox_dir.glob("*"):
            path.unlink()
        sandbox_dir.rmdir()
        for path in parent_dir.glob("*.json"):
            path.unlink()
        parent_dir.rmdir()
