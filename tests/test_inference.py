"""Checks for inference configuration and runtime setup."""

from __future__ import annotations

import json

import pytest

from inference import (
    BENCHMARK_NAME,
    build_inference_client,
    load_inference_config,
    log_end,
    log_start,
    log_step,
    run_all_inference,
    run_inference,
)
from server.agent_harness import HARNESS_BACKEND_NAME
from server.policy_agent import AGENT_NAME as POLICY_AGENT_NAME


def proposal_json(action: str) -> str:
    return json.dumps(
        {
            "action": action,
            "objective": "Tighten the target mismatch",
            "rationale": "Use the strongest evaluated move on the current board.",
            "expected_outcome": "Improve the tradeoff between error and cost.",
            "confidence": 0.8,
        }
    )


class FakeResponses:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [proposal_json("c_up")])
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.outputs:
            output_text = self.outputs.pop(0)
            self.outputs.append(output_text)
        else:
            output_text = proposal_json("c_up")
        return type("Response", (), {"output_text": output_text})()


class FakeClient:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.responses = FakeResponses(outputs=outputs)


def test_load_inference_config_reads_required_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    config = load_inference_config()

    assert config.api_base_url == "https://router.huggingface.co/v1"
    assert config.model_name == "test-model"
    assert config.api_key == "hf_test_token"


def test_load_inference_config_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY or HF_TOKEN is required"):
        load_inference_config()


def test_load_inference_config_accepts_openai_api_key_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_test_token")

    config = load_inference_config()

    assert config.api_key == "openai_test_token"


def test_build_inference_client_uses_env_config(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    config = load_inference_config()
    client = build_inference_client(config)

    assert str(client.base_url) == "https://router.huggingface.co/v1/"
    assert client.api_key == "hf_test_token"


def test_run_all_inference_uses_configured_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    fake_client = FakeClient(outputs=[proposal_json("c_up")])
    results = run_all_inference("tasks", client=fake_client)

    assert results
    assert all(result["details"]["model_name"] == "test-model" for result in results)
    assert all(result["details"]["agent_backend"] == HARNESS_BACKEND_NAME for result in results)
    assert fake_client.responses.calls
    assert fake_client.responses.calls[0]["model"] == "test-model"
    assert fake_client.responses.calls[0]["text"]["format"]["type"] == "json_schema"


def test_log_start_matches_required_format(capsys):
    log_start(task="lp_1khz_budget", env=BENCHMARK_NAME, model="test-model")

    assert capsys.readouterr().out == "[START] task=lp_1khz_budget env=circuitrl model=test-model\n"


def test_log_step_matches_required_format(capsys):
    log_step(step=3, action="r_up", reward=0.125, done=False, error=None)

    assert capsys.readouterr().out == "[STEP] step=3 action=r_up reward=0.12 done=false error=null\n"


def test_log_end_matches_required_format(capsys):
    log_end(success=True, steps=2, score=0.8754, rewards=[0.5, 0.375])

    assert capsys.readouterr().out == "[END] success=true steps=2 score=0.875 rewards=0.50,0.38\n"


def test_run_inference_emits_required_lines(monkeypatch, capsys):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    result = run_inference(
        "tasks/lp_1khz_budget.json",
        client=FakeClient(outputs=[proposal_json("c_up")]),
        log_stdout=True,
    )
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines[0] == "[START] task=lp_1khz_budget env=circuitrl model=test-model"
    assert lines[-1].startswith("[END] success=")
    assert sum(line.startswith("[STEP]") for line in lines) == result["details"]["steps"]
    assert len(lines) == result["details"]["steps"] + 2


def test_run_all_inference_emits_start_and_end_for_each_task(monkeypatch, capsys):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    results = run_all_inference(
        "tasks",
        client=FakeClient(outputs=[proposal_json("c_up")]),
        log_stdout=True,
    )
    lines = capsys.readouterr().out.strip().splitlines()

    assert len(results) == 4
    assert sum(line.startswith("[START]") for line in lines) == 4
    assert sum(line.startswith("[END]") for line in lines) == 4


def test_run_inference_defaults_to_llm_backend_when_env_present(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("API_BASE_URL", raising=False)

    result = run_inference(
        "tasks/lp_1khz_budget.json",
        client=FakeClient(outputs=[proposal_json("c_up")]),
    )

    assert result["task_id"] == "lp_1khz_budget"
    assert result["details"]["agent_backend"] == HARNESS_BACKEND_NAME
    assert result["details"]["model_name"] == "test-model"
    assert result["details"]["simulator_evaluations"] >= 4


def test_run_inference_default_llm_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY or HF_TOKEN is required"):
        run_inference("tasks/lp_1khz_budget.json")


def test_run_inference_policy_backend_remains_available(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)

    result = run_inference("tasks/lp_1khz_budget.json", agent_backend="policy")

    assert result["task_id"] == "lp_1khz_budget"
    assert result["details"]["agent_backend"] == "policy"
    assert result["details"]["model_name"] == POLICY_AGENT_NAME
    assert result["score"] >= 0.8
