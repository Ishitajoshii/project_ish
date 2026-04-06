"""Task loader for deterministic benchmark JSON definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_task(path: str | Path) -> dict[str, Any]:
    """Load and return one task definition as a dictionary."""

    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    return task
