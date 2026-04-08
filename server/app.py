"""Thin FastAPI/OpenEnv wrapper around the deterministic circuit runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from models import (
    CircuitAction,
    CircuitObservation,
    CircuitReward,
    CircuitState,
    CircuitStepInfo,
    CircuitTaskSpec,
)
from server.agent_harness import AgentHarness
from server.environment import CircuitEnvironment
from server.task_loader import TASKS_DIR, get_task_ids_in_order, load_tasks
from server.ui_service import UiCatalogResponse, UiPlaybackPayload, build_episode_payload, build_initial_payload, build_ui_catalog

APP_NAME = "circuitrl"
APP_DESCRIPTION = "OpenEnv benchmark for constrained RC circuit tuning with deterministic range-based tasks."
APP_VERSION = "0.1.0"


class ResetRequest(BaseModel):
    """Optional task selector for starting a fresh episode."""

    task_id: str | None = None


class StepResponse(BaseModel):
    """Step transition payload exposed by the API."""

    observation: CircuitObservation
    reward: CircuitReward
    done: bool
    info: CircuitStepInfo


class TasksResponse(BaseModel):
    """Deterministic task catalog for UI/debugging use."""

    task_ids: list[str]


class MetadataResponse(BaseModel):
    """OpenEnv metadata surfaced for validators and tooling."""

    name: str
    description: str
    version: str
    task_ids: list[str]
    default_task_id: str


class SchemaResponse(BaseModel):
    """JSON schema bundle for the typed OpenEnv models."""

    action: dict[str, Any]
    observation: dict[str, Any]
    reward: dict[str, Any]
    step_info: dict[str, Any]
    state: dict[str, Any]
    task: dict[str, Any]


class MpcLikeRequest(BaseModel):
    """Minimal JSON-RPC request envelope for OpenEnv validator compatibility."""

    jsonrpc: str | None = None
    id: str | int | None = None
    method: str | None = None
    params: dict[str, Any] | list[Any] | None = None


class McpResponse(BaseModel):
    """Minimal JSON-RPC response envelope for the validator."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, Any]


APP_TASKS_DIR = Path(TASKS_DIR)
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[1] / "ui" / "dist"
TASKS = load_tasks(APP_TASKS_DIR)
TASK_IDS = get_task_ids_in_order(TASKS)
DEFAULT_TASK_ID = TASK_IDS[0]
ENV = CircuitEnvironment(tasks=TASKS)

app = FastAPI(title="CircuitRL OpenEnv Server", version=APP_VERSION)


def build_ui_episode_agent(task_id: str) -> AgentHarness:
    """Build the model-driven episode agent for one UI task run."""

    return AgentHarness.from_env({task_id: TASKS[task_id]})


def _resolve_task_id(request: ResetRequest | None) -> str:
    if request is None or request.task_id is None:
        return DEFAULT_TASK_ID
    return request.task_id


@app.get("/health")
def health() -> dict[str, str]:
    """Cheap readiness probe for validators and deployment checks."""

    return {"status": "healthy"}


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

    task = ENV._require_task()
    success_threshold = task.success_tolerance_pct / 100.0
    terminated_by = (
        "success"
        if done and observation.normalized_error <= success_threshold
        else "max_steps"
        if done
        else "in_progress"
    )
    accuracy_score = max(0.0, 1.0 - observation.normalized_error)
    cost_efficiency = max(0.0, 1.0 - observation.current_cost)
    step_efficiency = max(0.0, 1.0 - (ENV.step_count / max(task.max_steps, 1)))

    return StepResponse(
        observation=observation,
        reward=CircuitReward(
            value=reward,
            accuracy_score=accuracy_score,
            cost_efficiency=cost_efficiency,
            step_efficiency=step_efficiency,
        ),
        done=done,
        info=CircuitStepInfo(
            task_id=task.task_id,
            step_count=ENV.step_count,
            best_score=ENV.score(),
            current_hz=observation.current_hz,
            normalized_error=observation.normalized_error,
            current_cost=observation.current_cost,
            success_threshold=success_threshold,
            terminated_by=terminated_by,
        ),
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


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Expose OpenEnv environment metadata for validators and local tooling."""

    return MetadataResponse(
        name=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        task_ids=TASK_IDS,
        default_task_id=DEFAULT_TASK_ID,
    )


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    """Expose the typed action/observation/state schemas for validators and tools."""

    return SchemaResponse(
        action=CircuitAction.model_json_schema(),
        observation=CircuitObservation.model_json_schema(),
        reward=CircuitReward.model_json_schema(),
        step_info=CircuitStepInfo.model_json_schema(),
        state=CircuitState.model_json_schema(),
        task=CircuitTaskSpec.model_json_schema(),
    )


@app.post("/mcp", response_model=McpResponse)
def mcp(request: MpcLikeRequest | None = Body(default=None)) -> McpResponse:
    """Return a minimal JSON-RPC response for OpenEnv MCP reachability checks."""

    method = request.method if request is not None else None
    request_id = request.id if request is not None else None

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": APP_NAME,
                "version": APP_VERSION,
            },
            "capabilities": {},
        }
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {"status": "ok"}

    return McpResponse(id=request_id, result=result)


@app.get("/api/ui/catalog", response_model=UiCatalogResponse)
def ui_catalog() -> UiCatalogResponse:
    """Expose the ordered task catalog and UI constants for the Vite app."""

    return build_ui_catalog(tasks=TASKS, task_ids=TASK_IDS, default_task_id=DEFAULT_TASK_ID)


@app.get("/api/ui/preview", response_model=UiPlaybackPayload)
def ui_preview(task_id: str | None = None) -> UiPlaybackPayload:
    """Build the single-frame preview payload for a selected task."""

    resolved_task_id = task_id or DEFAULT_TASK_ID
    try:
        return build_initial_payload(TASKS[resolved_task_id])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown task_id: {resolved_task_id}") from exc


@app.get("/api/ui/episode", response_model=UiPlaybackPayload)
def ui_episode(task_id: str | None = None) -> UiPlaybackPayload:
    """Build the full playback payload for a selected task."""

    resolved_task_id = task_id or DEFAULT_TASK_ID
    try:
        return build_episode_payload(
            TASKS[resolved_task_id],
            agent=build_ui_episode_agent(resolved_task_id),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown task_id: {resolved_task_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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


if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")


if __name__ == "__main__":
    main()
