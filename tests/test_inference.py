"""Checks for inference configuration and runtime setup."""

import pytest

from inference import build_openai_client, load_inference_config, run_all_inference


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
