"""FastAPI server exposing reset/step/score endpoints in OpenEnv style."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import CircuitAction
from server.environment import CircuitEnvironment
from server.task_loader import load_task

app = FastAPI(title="circuitrl-openenv")
_ENV: CircuitEnvironment | None = None


class ResetRequest(BaseModel):
    """Request body for selecting a task file and resetting state."""

    task_path: str


@app.get("/health")
def health() -> dict[str, str]:
    """Basic readiness probe for local orchestration and CI smoke checks."""

    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """Load task and create a fresh environment state."""

    global _ENV
    task = load_task(req.task_path)
    _ENV = CircuitEnvironment({task.task_id: task})
    obs = _ENV.reset(task.task_id)
    return {"observation": obs.model_dump(), "task_id": task.task_id}


@app.post("/step")
def step(action: CircuitAction) -> dict:
    """Apply one action transition and return the next observation."""

    if _ENV is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    obs, reward, done = _ENV.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done}


@app.get("/score")
def score() -> dict:
    """Return normalized score for current trajectory in [0, 1]."""

    if _ENV is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    return {"score": _ENV.score()}
