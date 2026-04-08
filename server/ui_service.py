"""Typed UI payload builders for the Vite frontend."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from models import CircuitAction, CircuitObservation, CircuitTaskSpec
from server.agent_harness import AgentHarness, EpisodeRunResult
from server.baselines import run_bruteforce_baseline, run_heuristic_baseline, run_random_baseline
from server.environment import CircuitEnvironment
from server.grader import is_success
from server.policy_agent import TabularValueIterationAgent, run_policy_episode
from server.simulator import ACTION_SCALE_FACTOR, valid_actions

AGENT_LABEL = "CircuitRL agent"
ACTION_LABELS = {
    "init": "Bench Reset",
    "r_up": "Increase R",
    "r_down": "Decrease R",
    "c_up": "Increase C",
    "c_down": "Decrease C",
}
ACTION_SYMBOLS = {
    "init": "INIT",
    "r_up": "R+",
    "r_down": "R-",
    "c_up": "C+",
    "c_down": "C-",
}


class PlaybackFrame(BaseModel):
    """One chart-friendly step snapshot."""

    step: int
    action: str
    action_label: str
    action_symbol: str
    reward: float | None = None
    best_score: float
    note: str
    current_r_ohms: float
    current_c_farads: float
    current_hz: float
    normalized_error: float
    current_cost: float
    remaining_steps: int
    delta_hz: float
    within_tolerance: bool


class EpisodeSummary(BaseModel):
    """Best-state summary shown after playback completes."""

    score: float
    success: bool
    steps_used: int
    achieved_hz: float
    best_error: float
    best_cost: float
    best_r_ohms: float
    best_c_farads: float


class BaselineComparison(BaseModel):
    """One row in the baseline comparison table."""

    baseline_name: str
    task_id: str
    label: str
    score: float
    success: bool
    steps_used: int
    evaluations: int | None = None
    achieved_hz: float
    current_r_ohms: float
    current_c_farads: float
    normalized_error: float
    normalized_cost: float


class UiPlaybackPayload(BaseModel):
    """Shared payload shape for preview and full episode playback."""

    task: CircuitTaskSpec
    frames: list[PlaybackFrame]
    summary: EpisodeSummary | None = None
    comparisons: list[BaselineComparison] = Field(default_factory=list)


class UiCatalogResponse(BaseModel):
    """Boot payload for the Vite app."""

    default_task_id: str
    task_ids: list[str]
    tasks: list[CircuitTaskSpec]
    action_scale_factor: float


def build_ui_catalog(
    *,
    tasks: Mapping[str, CircuitTaskSpec],
    task_ids: Sequence[str],
    default_task_id: str,
) -> UiCatalogResponse:
    """Return the task catalog in the deterministic UI order."""

    return UiCatalogResponse(
        default_task_id=default_task_id,
        task_ids=list(task_ids),
        tasks=[tasks[task_id] for task_id in task_ids],
        action_scale_factor=ACTION_SCALE_FACTOR,
    )


def build_initial_payload(task: CircuitTaskSpec) -> UiPlaybackPayload:
    """Build the initial task preview before episode playback begins."""

    env = CircuitEnvironment({task.task_id: task})
    observation = env.reset(task.task_id)
    return UiPlaybackPayload(
        task=task,
        frames=[
            _build_frame(
                task,
                observation,
                step=0,
                action="init",
                reward=None,
                best_score=0.0,
                note="Bench staged. The controller has the target spec, the starting RC pair, and a full step budget.",
            )
        ],
    )


def build_episode_payload(
    task: CircuitTaskSpec,
    *,
    agent: AgentHarness | TabularValueIterationAgent,
) -> UiPlaybackPayload:
    """Run one demo episode and capture the full trajectory."""

    if isinstance(agent, AgentHarness):
        run_result = agent.run_episode(CircuitEnvironment({task.task_id: task}), task.task_id)
        frames = _build_frames_from_llm_result(task, run_result)
        state = run_result.state
        agent_score = run_result.score
        agent_label = f"{run_result.model_name} agent"
        agent_comparison = BaselineComparison.model_validate(
            {
                "baseline_name": "agent",
                "task_id": task.task_id,
                "label": agent_label,
                "score": run_result.score,
                "success": run_result.success,
                "steps_used": run_result.state.step_count,
                "evaluations": run_result.simulator_evaluations,
                "achieved_hz": float(run_result.state.best_hz),
                "current_r_ohms": float(run_result.state.best_r_ohms),
                "current_c_farads": float(run_result.state.best_c_farads),
                "normalized_error": float(run_result.state.best_normalized_error),
                "normalized_cost": float(run_result.state.best_normalized_cost),
            }
        )
        final_observation = run_result.final_observation
    else:
        resolved_agent = agent
        env = CircuitEnvironment({task.task_id: task})
        observation = env.reset(task.task_id)
        frames = [
            _build_frame(
                task,
                observation,
                step=0,
                action="init",
                reward=None,
                best_score=0.0,
                note="Bench staged. The controller reads the mismatch, then scores all four legal moves against the real reward.",
            )
        ]
        while not env.is_done:
            action = resolved_agent.choose_action(observation, env.score())
            previous_observation = observation
            observation, reward, done = env.step(CircuitAction(action=action))
            frames.append(
                _build_frame(
                    task,
                    observation,
                    step=env.step_count,
                    action=action,
                    reward=reward,
                    best_score=env.score(),
                    note=_describe_transition(task, previous_observation, observation, action),
                )
            )
            if done:
                break
        state = env.state()
        agent_score = float(env.score())
        agent_label = AGENT_LABEL
        agent_comparison = BaselineComparison.model_validate(
            {
                **run_policy_episode(
                    CircuitEnvironment({task.task_id: task}),
                    task.task_id,
                    resolved_agent,
                    baseline_name="agent",
                ),
                "label": AGENT_LABEL,
                "evaluations": state.step_count * len(valid_actions()),
            }
        )
        final_observation = observation

    comparisons = [
        agent_comparison,
        BaselineComparison.model_validate(
            {
                **run_random_baseline(CircuitEnvironment({task.task_id: task}), task.task_id, seed=7),
                "label": "Random",
            }
        ),
        BaselineComparison.model_validate(
            {
                **run_heuristic_baseline(CircuitEnvironment({task.task_id: task}), task.task_id),
                "label": "Heuristic",
            }
        ),
        BaselineComparison.model_validate(
            {
                **run_bruteforce_baseline(task),
                "label": "Brute-force",
            }
        ),
    ]
    return UiPlaybackPayload(
        task=task,
        frames=frames,
        summary=EpisodeSummary(
            score=agent_score,
            success=is_success(agent_score),
            steps_used=state.step_count,
            achieved_hz=float(state.best_hz or final_observation.current_hz or 0.0),
            best_error=float(state.best_normalized_error or final_observation.normalized_error),
            best_cost=float(state.best_normalized_cost or final_observation.current_cost),
            best_r_ohms=float(state.best_r_ohms or final_observation.current_r_ohms),
            best_c_farads=float(state.best_c_farads or final_observation.current_c_farads),
        ),
        comparisons=comparisons,
    )


def _build_frames_from_llm_result(
    task: CircuitTaskSpec,
    run_result: EpisodeRunResult,
) -> list[PlaybackFrame]:
    """Convert one LLM-backed episode run into playback frames."""

    start_observation = CircuitEnvironment({task.task_id: task}).reset(task.task_id)
    frames = [
        _build_frame(
            task,
            start_observation,
            step=0,
            action="init",
            reward=None,
            best_score=0.0,
            note="Bench staged. The harness asks the model to pick from an exact evaluator board for each legal move.",
        )
    ]
    for trace_step in run_result.trace_steps:
        frames.append(
            PlaybackFrame(
                step=trace_step.step,
                action=trace_step.action,
                action_label=ACTION_LABELS[trace_step.action],
                action_symbol=ACTION_SYMBOLS[trace_step.action],
                reward=trace_step.reward,
                best_score=trace_step.best_score_after,
                note=trace_step.note,
                current_r_ohms=trace_step.current_r_ohms,
                current_c_farads=trace_step.current_c_farads,
                current_hz=trace_step.current_hz,
                normalized_error=trace_step.normalized_error,
                current_cost=trace_step.normalized_cost,
                remaining_steps=trace_step.remaining_steps,
                delta_hz=trace_step.current_hz - task.target_hz,
                within_tolerance=trace_step.normalized_error <= (task.success_tolerance_pct / 100.0),
            )
        )
    return frames


def _build_frame(
    task: CircuitTaskSpec,
    observation: CircuitObservation,
    *,
    step: int,
    action: str,
    reward: float | None,
    best_score: float,
    note: str,
) -> PlaybackFrame:
    """Convert one observation into a chart-friendly playback frame."""

    return PlaybackFrame(
        step=step,
        action=action,
        action_label=ACTION_LABELS[action],
        action_symbol=ACTION_SYMBOLS[action],
        reward=reward,
        best_score=best_score,
        note=note,
        current_r_ohms=observation.current_r_ohms,
        current_c_farads=observation.current_c_farads,
        current_hz=observation.current_hz,
        normalized_error=observation.normalized_error,
        current_cost=observation.current_cost,
        remaining_steps=observation.remaining_steps,
        delta_hz=observation.current_hz - task.target_hz,
        within_tolerance=observation.normalized_error <= (task.success_tolerance_pct / 100.0),
    )


def _describe_transition(
    task: CircuitTaskSpec,
    previous: CircuitObservation,
    current: CircuitObservation,
    action: str,
) -> str:
    """Build a short engineering narrative for one control step."""

    error_before = previous.normalized_error
    error_after = current.normalized_error
    direction = "pull the cutoff down" if action in {"r_up", "c_up"} else "push the cutoff up"
    component = "resistor" if action.startswith("r") else "capacitor"
    improvement = error_before - error_after
    if improvement > 1e-9:
        outcome = (
            f"Error tightens from {error_before * 100:.1f}% to {error_after * 100:.1f}%, "
            "so the move pays off immediately."
        )
    elif improvement < -1e-9:
        outcome = (
            f"Error widens to {error_after * 100:.1f}%, but the controller keeps the move because "
            "the reward still balances cost and remaining step budget."
        )
    else:
        outcome = "The move keeps error flat and mainly shifts the cost profile."
    return f"The controller changes the {component} to {direction}. {outcome}"
