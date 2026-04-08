"""Checks for the model-driven evaluator harness."""

from __future__ import annotations

import json

from server.environment import CircuitEnvironment
from server.agent_harness import (
    AgentHarness,
    HarnessConfig,
    evaluate_candidate_actions,
)
from server.task_loader import load_tasks


def proposal_json(action: str) -> str:
    return json.dumps(
        {
            "action": action,
            "objective": "Tighten the target mismatch",
            "rationale": "Choose the best tradeoff from the evaluator board.",
            "expected_outcome": "Improve score or preserve the strongest score so far.",
            "confidence": 0.8,
        }
    )


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text=output_text)


def test_evaluate_candidate_actions_returns_all_legal_actions():
    task = load_tasks()["lp_1khz_budget"]
    env = CircuitEnvironment({task.task_id: task})
    observation = env.reset(task.task_id)

    candidates = evaluate_candidate_actions(task, observation, best_score=env.score())

    assert set(candidates) == {"r_up", "r_down", "c_up", "c_down"}
    assert max(candidates.values(), key=lambda candidate: candidate.reward).action == "c_up"


def test_harness_accepts_model_action_when_it_matches_the_evaluator_board():
    tasks = load_tasks()
    task = tasks["lp_1khz_budget"]
    agent = AgentHarness(
        tasks={task.task_id: task},
        config=HarnessConfig(
            api_base_url="https://api.openai.com/v1",
            model_name="test-model",
            api_key="test-key",
            max_revision_rounds=1,
        ),
        client=FakeClient(proposal_json("c_up")),
    )

    result = agent.run_episode(CircuitEnvironment({task.task_id: task}), task.task_id)

    assert result.trace_steps
    assert result.trace_steps[0].selected_by == "model"
    assert result.trace_steps[0].model_action == "c_up"
    assert result.trace_steps[0].action == "c_up"
    assert result.score > 0.0


def test_harness_overrides_a_dominated_model_action_after_revisions():
    tasks = load_tasks()
    task = tasks["lp_1khz_budget"]
    agent = AgentHarness(
        tasks={task.task_id: task},
        config=HarnessConfig(
            api_base_url="https://api.openai.com/v1",
            model_name="test-model",
            api_key="test-key",
            max_revision_rounds=1,
            override_margin=0.0,
        ),
        client=FakeClient(proposal_json("r_up")),
    )

    result = agent.run_episode(CircuitEnvironment({task.task_id: task}), task.task_id)

    assert result.trace_steps
    assert result.trace_steps[0].selected_by == "evaluator_override"
    assert result.trace_steps[0].model_action == "r_up"
    assert result.trace_steps[0].action == "c_up"
    assert result.simulator_evaluations >= 4
