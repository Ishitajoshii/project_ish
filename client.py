"""Local/OpenEnv client wrapper for interacting with the circuitrl server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class OpenEnvClient:
    """Tiny HTTP client wrapper with reset/step helpers for local development."""

    base_url: str = "http://127.0.0.1:8000"

    def reset(self, task_path: str) -> dict[str, Any]:
        resp = httpx.post(f"{self.base_url}/reset", json={"task_path": task_path}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = httpx.post(f"{self.base_url}/step", json=action, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def score(self) -> dict[str, Any]:
        resp = httpx.get(f"{self.base_url}/score", timeout=10)
        resp.raise_for_status()
        return resp.json()
