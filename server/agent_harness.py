"""Generic agent harness for model-driven circuit tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from models import CircuitAction, CircuitActionType, CircuitObservation, CircuitState, CircuitTaskSpec
from server.grader import is_success
from server.simulator import (
    ACTION_SCALE_FACTOR,
    SUCCESS_TOLERANCE,
    apply_action,
    evaluate_circuit_state,
    valid_actions,
)

load_dotenv()

HARNESS_BACKEND_NAME = "model_eval_harness"
DEFAULT_MODEL_NAME = "configured-model"
ACTION_PRIORITY = {"r_down": 4, "c_down": 3, "r_up": 2, "c_up": 1}


@dataclass(frozen=True)
class HarnessConfig:
    """Runtime configuration for the model-driven tuning harness."""

    api_base_url: str | None
    model_name: str
    api_key: str
    reasoning_effort: str = "high"
    max_revision_rounds: int = 2
    override_margin: float = 0.03
    max_output_tokens: int = 300


class HarnessProposal(BaseModel):
    """Structured model proposal returned by the inference client."""

    model_config = ConfigDict(extra="forbid")

    action: CircuitActionType
    objective: str = Field(min_length=1, max_length=120)
    rationale: str = Field(min_length=1, max_length=280)
    expected_outcome: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Exact simulator-backed evaluation for one legal next action."""

    action: str
    next_r_ohms: float
    next_c_farads: float
    next_hz: float
    normalized_error: float
    normalized_cost: float
    reward: float
    best_score_after: float
    action_error: str | None = None


@dataclass(frozen=True)
class EpisodeTraceStep:
    """One executed decision and its surrounding evaluator context."""

    step: int
    action: str
    model_action: str
    selected_by: Literal["model", "evaluator_override"]
    reward: float
    best_score_after: float
    current_r_ohms: float
    current_c_farads: float
    current_hz: float
    normalized_error: float
    normalized_cost: float
    remaining_steps: int
    note: str
    critique: str | None
    revision_count: int
    best_available_action: str
    best_available_reward: float
    best_available_score: float
    candidate_evaluations: int


@dataclass(frozen=True)
class EpisodeRunResult:
    """Full episode result for inference and UI playback."""

    task_id: str
    model_name: str
    score: float
    success: bool
    state: CircuitState
    final_observation: CircuitObservation
    trace_steps: list[EpisodeTraceStep]
    simulator_evaluations: int


@dataclass
class AgentHarness:
    """Model-driven controller that uses exact simulator evals to refine actions."""

    tasks: Mapping[str, CircuitTaskSpec]
    config: HarnessConfig
    client: Any
    recent_history_limit: int = 4
    _schema: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._schema = HarnessProposal.model_json_schema()

    @classmethod
    def from_env(
        cls,
        tasks: Mapping[str, CircuitTaskSpec],
        *,
        client: Any | None = None,
        config: HarnessConfig | None = None,
    ) -> "AgentHarness":
        """Build a harness from environment configuration."""

        resolved_config = config or load_harness_config()
        resolved_client = client or build_model_client(resolved_config)
        return cls(tasks=tasks, config=resolved_config, client=resolved_client)

    @property
    def agent_name(self) -> str:
        return self.config.model_name

    def run_episode(
        self,
        env,
        task_id: str,
    ) -> EpisodeRunResult:
        """Run one full task using the model-driven evaluator loop."""

        task = self.tasks[task_id]
        observation = env.reset(task_id)
        trace_steps: list[EpisodeTraceStep] = []
        simulator_evaluations = 0

        while not env.is_done:
            previous_observation = observation
            best_score_before = env.score()
            decision = self._choose_action(
                task=task,
                observation=observation,
                best_score=best_score_before,
                history=trace_steps,
            )
            simulator_evaluations += decision["candidate_evaluations"]
            observation, reward, done = env.step(CircuitAction(action=decision["action"]))
            trace_steps.append(
                EpisodeTraceStep(
                    step=env.step_count,
                    action=decision["action"],
                    model_action=decision["model_action"],
                    selected_by=decision["selected_by"],
                    reward=reward,
                    best_score_after=env.score(),
                    current_r_ohms=observation.current_r_ohms,
                    current_c_farads=observation.current_c_farads,
                    current_hz=observation.current_hz,
                    normalized_error=observation.normalized_error,
                    normalized_cost=observation.current_cost,
                    remaining_steps=observation.remaining_steps,
                    note=self._build_step_note(
                        previous_observation=previous_observation,
                        current_observation=observation,
                        decision=decision,
                    ),
                    critique=decision["critique"],
                    revision_count=decision["revision_count"],
                    best_available_action=decision["best_candidate"].action,
                    best_available_reward=decision["best_candidate"].reward,
                    best_available_score=decision["best_candidate"].best_score_after,
                    candidate_evaluations=decision["candidate_evaluations"],
                )
            )
            if done:
                break

        state = env.state()
        score = env.score()
        return EpisodeRunResult(
            task_id=task_id,
            model_name=self.config.model_name,
            score=score,
            success=is_success(score),
            state=state,
            final_observation=observation,
            trace_steps=trace_steps,
            simulator_evaluations=simulator_evaluations,
        )

    def _choose_action(
        self,
        *,
        task: CircuitTaskSpec,
        observation: CircuitObservation,
        best_score: float,
        history: list[EpisodeTraceStep],
    ) -> dict[str, Any]:
        """Choose one action with critique-and-revise iterations."""

        candidate_map = evaluate_candidate_actions(task, observation, best_score)
        best_candidate = max(candidate_map.values(), key=_candidate_sort_key)
        critique: str | None = None
        last_error: str | None = None
        proposal: HarnessProposal | None = None

        for revision_index in range(self.config.max_revision_rounds + 1):
            try:
                proposal = self._request_proposal(
                    task=task,
                    observation=observation,
                    best_score=best_score,
                    history=history,
                    candidates=candidate_map,
                    critique=critique,
                    revision_index=revision_index,
                )
                proposal_candidate = candidate_map[proposal.action.value]
                if self._is_candidate_acceptable(proposal_candidate, best_candidate):
                    return {
                        "action": proposal.action.value,
                        "model_action": proposal.action.value,
                        "selected_by": "model",
                        "proposal": proposal,
                        "best_candidate": best_candidate,
                        "critique": critique,
                        "revision_count": revision_index,
                        "candidate_evaluations": len(candidate_map),
                    }
                critique = build_revision_feedback(
                    proposed_action=proposal.action.value,
                    proposed_candidate=proposal_candidate,
                    best_candidate=best_candidate,
                    revision_index=revision_index + 1,
                )
                last_error = None
            except Exception as exc:
                last_error = str(exc)
                critique = (
                    "Your previous response was invalid. "
                    f"Return strict JSON that matches the schema and choose only one legal action. Error: {last_error}"
                )

        selected_model_action = proposal.action.value if proposal is not None else best_candidate.action
        return {
            "action": best_candidate.action,
            "model_action": selected_model_action,
            "selected_by": "evaluator_override",
            "proposal": proposal,
            "best_candidate": best_candidate,
            "critique": critique or last_error,
            "revision_count": self.config.max_revision_rounds,
            "candidate_evaluations": len(candidate_map),
        }

    def _request_proposal(
        self,
        *,
        task: CircuitTaskSpec,
        observation: CircuitObservation,
        best_score: float,
        history: list[EpisodeTraceStep],
        candidates: dict[str, CandidateEvaluation],
        critique: str | None,
        revision_index: int,
    ) -> HarnessProposal:
        """Call the inference client and parse one structured action proposal."""

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_system_prompt(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._build_user_prompt(
                                task=task,
                                observation=observation,
                                best_score=best_score,
                                history=history,
                                candidates=candidates,
                                critique=critique,
                                revision_index=revision_index,
                            ),
                        }
                    ],
                },
            ],
            "max_output_tokens": self.config.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "circuit_action_proposal",
                    "schema": self._schema,
                    "strict": True,
                }
            },
            "temperature": 0.0,
        }
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}

        response = self.client.responses.create(**payload)
        output_text = extract_response_text(response)
        return HarnessProposal.model_validate_json(output_text)

    def _build_user_prompt(
        self,
        *,
        task: CircuitTaskSpec,
        observation: CircuitObservation,
        best_score: float,
        history: list[EpisodeTraceStep],
        candidates: dict[str, CandidateEvaluation],
        critique: str | None,
        revision_index: int,
    ) -> str:
        """Build the current decision context for the agent."""

        step_count = task.max_steps - observation.remaining_steps
        candidate_lines = []
        for candidate in sorted(candidates.values(), key=_candidate_sort_key, reverse=True):
            candidate_lines.append(
                (
                    f"- {candidate.action}: next_hz={candidate.next_hz:.3f}, "
                    f"error={candidate.normalized_error:.6f}, cost={candidate.normalized_cost:.6f}, "
                    f"reward={candidate.reward:.6f}, best_score_after={candidate.best_score_after:.6f}"
                )
            )
        history_lines = format_recent_history(history[-self.recent_history_limit :])
        critique_block = critique or "No critique yet. Pick the strongest move from the evaluator board."
        return (
            f"revision_index={revision_index}\n"
            f"task_id={task.task_id}\n"
            f"circuit_type={task.circuit_type}\n"
            f"target_hz={task.target_hz}\n"
            f"max_steps={task.max_steps}\n"
            f"success_tolerance_pct={task.success_tolerance_pct}\n"
            f"current_step={step_count}\n"
            f"remaining_steps={observation.remaining_steps}\n"
            f"best_score_so_far={best_score:.6f}\n"
            f"current_r_ohms={observation.current_r_ohms}\n"
            f"current_c_farads={observation.current_c_farads}\n"
            f"current_hz={observation.current_hz}\n"
            f"current_normalized_error={observation.normalized_error}\n"
            f"current_cost={observation.current_cost}\n"
            f"last_action_error={observation.last_action_error or 'null'}\n\n"
            f"Exact evaluator scoreboard for the next move:\n{chr(10).join(candidate_lines)}\n\n"
            f"Recent trajectory:\n{history_lines}\n\n"
            f"Revision feedback:\n{critique_block}"
        )

    def _is_candidate_acceptable(
        self,
        candidate: CandidateEvaluation,
        best_candidate: CandidateEvaluation,
    ) -> bool:
        """Decide whether the model's proposed action is good enough to accept."""

        if candidate.best_score_after + 1e-12 < best_candidate.best_score_after:
            return False
        return (best_candidate.reward - candidate.reward) <= self.config.override_margin

    @staticmethod
    def _build_step_note(
        *,
        previous_observation: CircuitObservation,
        current_observation: CircuitObservation,
        decision: dict[str, Any],
    ) -> str:
        """Build a short note for UI playback from the executed step."""

        direction = "down" if decision["action"] in {"r_up", "c_up"} else "up"
        component = "resistor" if decision["action"].startswith("r") else "capacitor"
        lead = f"The harness moved the {component}, aiming to pull the cutoff {direction}."
        result = (
            f" Error is now {current_observation.normalized_error * 100:.1f}% "
            f"with score {decision['best_candidate'].best_score_after:.3f} available."
        )
        if decision["selected_by"] == "evaluator_override":
            return (
                f"{lead} The model proposed {decision['model_action']}, but the verifier executed "
                f"{decision['action']} because it dominated the current revision board.{result}"
            )
        return f"{lead} The model kept control after evaluator review.{result}"


def load_harness_config() -> HarnessConfig:
    """Read runtime configuration while preserving evaluator env compatibility."""

    api_base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME
    if api_base_url is None or "api.openai.com" in api_base_url:
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("HF_TOKEN")
        )
    else:
        api_key = (
            os.getenv("HF_TOKEN")
            or os.getenv("API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_API_KEY")
        )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN is required for the model harness")

    return HarnessConfig(
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
        reasoning_effort=os.getenv("AGENT_REASONING_EFFORT") or "high",
        max_revision_rounds=int(os.getenv("AGENT_MAX_REVISIONS") or "2"),
        override_margin=float(os.getenv("AGENT_OVERRIDE_MARGIN") or "0.03"),
        max_output_tokens=int(os.getenv("AGENT_MAX_OUTPUT_TOKENS") or "300"),
    )


def build_model_client(config: HarnessConfig) -> OpenAI:
    """Create the OpenAI client used by the harness."""

    kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.api_base_url:
        kwargs["base_url"] = config.api_base_url
    return OpenAI(**kwargs)


def run_harness_episode(
    env,
    task_id: str,
    agent: AgentHarness,
    *,
    baseline_name: str = "agent",
) -> dict[str, Any]:
    """Run one shared baseline-style episode using the harness."""

    result = agent.run_episode(env, task_id)
    return {
        "baseline_name": baseline_name,
        "task_id": task_id,
        "score": result.score,
        "success": result.success,
        "steps_used": result.state.step_count,
        "evaluations": result.simulator_evaluations,
        "achieved_hz": float(result.state.best_hz),
        "current_r_ohms": float(result.state.best_r_ohms),
        "current_c_farads": float(result.state.best_c_farads),
        "normalized_error": float(result.state.best_normalized_error),
        "normalized_cost": float(result.state.best_normalized_cost),
    }


def build_system_prompt() -> str:
    """Return the fixed system prompt for the tuning harness."""

    return (
        "You are CircuitRL, an autonomous analog circuit tuning agent. "
        "Your job is to choose one legal RC action per step to maximize the benchmark score. "
        "The final episode score is the best reward seen at any step, not just the final state. "
        "Use the evaluator scoreboard as ground truth for the next-step consequences of each legal action. "
        "If revision feedback says your prior suggestion was dominated, revise decisively. "
        "Only use these legal actions: r_up, r_down, c_up, c_down. "
        "Physics reminder: increasing R or C lowers cutoff frequency; decreasing R or C raises cutoff frequency. "
        "Return strict JSON that matches the provided schema."
    )


def evaluate_candidate_actions(
    task: CircuitTaskSpec,
    observation: CircuitObservation,
    best_score: float,
) -> dict[str, CandidateEvaluation]:
    """Evaluate all legal next actions from the current observation."""

    step_count = task.max_steps - observation.remaining_steps
    candidate_map: dict[str, CandidateEvaluation] = {}
    for action in valid_actions():
        next_r_ohms, next_c_farads, action_error = apply_action(
            observation.current_r_ohms,
            observation.current_c_farads,
            action,
            ACTION_SCALE_FACTOR,
            task.min_r_ohms,
            task.max_r_ohms,
            task.min_c_farads,
            task.max_c_farads,
        )
        metrics = evaluate_circuit_state(
            r_ohms=next_r_ohms,
            c_farads=next_c_farads,
            target_hz=task.target_hz,
            step_count=step_count + 1,
            max_steps=task.max_steps,
            success_tolerance=SUCCESS_TOLERANCE,
            min_r_ohms=task.min_r_ohms,
            max_r_ohms=task.max_r_ohms,
            min_c_farads=task.min_c_farads,
            max_c_farads=task.max_c_farads,
        )
        reward = float(metrics["reward"])
        candidate_map[action] = CandidateEvaluation(
            action=action,
            next_r_ohms=next_r_ohms,
            next_c_farads=next_c_farads,
            next_hz=float(metrics["current_hz"]),
            normalized_error=float(metrics["normalized_error"]),
            normalized_cost=float(metrics["normalized_cost"]),
            reward=reward,
            best_score_after=max(best_score, reward),
            action_error=action_error,
        )
    return candidate_map


def build_revision_feedback(
    *,
    proposed_action: str,
    proposed_candidate: CandidateEvaluation,
    best_candidate: CandidateEvaluation,
    revision_index: int,
) -> str:
    """Summarize why the previous proposal should be revised."""

    return (
        f"Revision {revision_index}: your proposal {proposed_action} yields reward "
        f"{proposed_candidate.reward:.6f} and best_score_after {proposed_candidate.best_score_after:.6f}. "
        f"The strongest current move is {best_candidate.action} with reward {best_candidate.reward:.6f} "
        f"and best_score_after {best_candidate.best_score_after:.6f}. "
        "Revise toward the better engineering tradeoff unless you can justify a superior long-horizon reason."
    )


def format_recent_history(history: list[EpisodeTraceStep]) -> str:
    """Render recent step history compactly for the next decision prompt."""

    if not history:
        return "No live steps yet."
    lines = []
    for step in history:
        lines.append(
            (
                f"- step={step.step}, executed={step.action}, selected_by={step.selected_by}, "
                f"reward={step.reward:.6f}, best_score_after={step.best_score_after:.6f}, "
                f"error={step.normalized_error:.6f}, cost={step.normalized_cost:.6f}"
            )
        )
    return "\n".join(lines)


def extract_response_text(response: Any) -> str:
    """Extract text from a Responses API result or a test double."""

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    if isinstance(response, dict):
        dict_output_text = response.get("output_text")
        if isinstance(dict_output_text, str) and dict_output_text.strip():
            return dict_output_text

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if output is None:
        raise ValueError("response did not contain output_text")

    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if content is None:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)

    if parts:
        return "".join(parts)
    raise ValueError("response did not contain output text content")


def _candidate_sort_key(candidate: CandidateEvaluation) -> tuple[float, float, float, float, int]:
    """Stable sort key for immediate evaluator boards."""

    return (
        candidate.best_score_after,
        candidate.reward,
        -candidate.normalized_error,
        -candidate.normalized_cost,
        ACTION_PRIORITY[candidate.action],
    )
