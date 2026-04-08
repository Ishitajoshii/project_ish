"""Thin FastAPI/OpenEnv wrapper around the deterministic circuit runtime."""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from models import CircuitAction, CircuitObservation, CircuitState
from server.environment import CircuitEnvironment
from server.task_loader import TASKS_DIR, get_task_ids_in_order, load_tasks


class ResetRequest(BaseModel):
    """Optional task selector for starting a fresh episode."""

    task_id: str | None = None


class StepResponse(BaseModel):
    """Step transition payload exposed by the API."""

    observation: CircuitObservation
    reward: float
    done: bool


class TasksResponse(BaseModel):
    """Deterministic task catalog for UI/debugging use."""

    task_ids: list[str]


APP_TASKS_DIR = Path(TASKS_DIR)
TASKS = load_tasks(APP_TASKS_DIR)
TASK_IDS = get_task_ids_in_order(TASKS)
DEFAULT_TASK_ID = TASK_IDS[0]
ENV = CircuitEnvironment(tasks=TASKS)

app = FastAPI(title="CircuitRL OpenEnv Server")


def _resolve_task_id(request: ResetRequest | None) -> str:
    if request is None or request.task_id is None:
        return DEFAULT_TASK_ID
    return request.task_id


@app.get("/health")
def health() -> dict[str, str]:
    """Cheap readiness probe for validators and deployment checks."""

    return {"status": "ok"}


@app.post("/reset", response_model=CircuitObservation)
def reset(request: ResetRequest | None = Body(default=None)) -> CircuitObservation:
    """Reset the shared environment to the selected or default benchmark task."""

    task_id = _resolve_task_id(request)
    try:
        return ENV.reset(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: CircuitAction) -> StepResponse:
    """Apply one action and return the resulting observation and reward."""

    try:
        observation, reward, done = ENV.step(action)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
    )


@app.get("/state", response_model=CircuitState)
def state() -> CircuitState:
    """Return the current episode summary."""

    try:
        return ENV.state()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    """Expose the deterministic task identifiers for UI/debugging use."""

    return TasksResponse(task_ids=TASK_IDS)


@app.get("/score")
def score() -> dict[str, float]:
    """Backward-compatible score endpoint for local tooling."""

    try:
        return {"score": ENV.score()}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    """Run the FastAPI app with uvicorn for local/dev or validator entrypoints."""

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
