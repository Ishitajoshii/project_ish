"""Checks for inference configuration and runtime setup."""

import pytest

from inference import (
    BENCHMARK_NAME,
    build_openai_client,
    load_inference_config,
    log_end,
    log_start,
    log_step,
    run_all_inference,
    run_inference,
)


def test_load_inference_config_reads_required_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    config = load_inference_config()

    assert config.api_base_url == "https://router.huggingface.co/v1"
    assert config.model_name == "test-model"
    assert config.hf_token == "hf_test_token"


def test_load_inference_config_rejects_missing_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN is required for inference"):
        load_inference_config()


def test_build_openai_client_uses_env_config(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    config = load_inference_config()
    client = build_openai_client(config)

    assert str(client.base_url) == "https://router.huggingface.co/v1/"
    assert client.api_key == "hf_test_token"


def test_run_all_inference_uses_configured_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    results = run_all_inference("tasks")

    assert results
    assert all(result["details"]["model_name"] == "test-model" for result in results)


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

    result = run_inference("tasks/lp_1khz_budget.json", log_stdout=True)
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines[0] == "[START] task=lp_1khz_budget env=circuitrl model=test-model"
    assert lines[-1].startswith("[END] success=")
    assert sum(line.startswith("[STEP]") for line in lines) == result["details"]["steps"]
    assert len(lines) == result["details"]["steps"] + 2


def test_run_all_inference_emits_start_and_end_for_each_task(monkeypatch, capsys):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    results = run_all_inference("tasks", log_stdout=True)
    lines = capsys.readouterr().out.strip().splitlines()

    assert len(results) == 4
    assert sum(line.startswith("[START]") for line in lines) == 4
    assert sum(line.startswith("[END]") for line in lines) == 4
