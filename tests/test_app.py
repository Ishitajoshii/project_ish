"""Smoke checks for the FastAPI/OpenEnv wrapper."""

from fastapi.testclient import TestClient

from server.app import DEFAULT_TASK_ID, ENV, TASK_IDS, TASKS, app
from server.policy_agent import TabularValueIterationAgent


client = TestClient(app)


def test_health_returns_ok():
    ENV.close()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_reset_without_body_uses_default_task():
    ENV.close()
    response = client.post("/reset", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == DEFAULT_TASK_ID
    assert payload["remaining_steps"] > 0


def test_reset_unknown_task_returns_404():
    ENV.close()
    response = client.post("/reset", json={"task_id": "does_not_exist"})

    assert response.status_code == 404
    assert "unknown task_id" in response.json()["detail"]


def test_step_before_reset_returns_400():
    ENV.close()
    response = client.post("/step", json={"action": "r_up"})

    assert response.status_code == 400
    assert "environment must be reset before use" in response.json()["detail"]


def test_state_before_reset_returns_400():
    ENV.close()
    response = client.get("/state")

    assert response.status_code == 400
    assert "environment must be reset before use" in response.json()["detail"]


def test_step_and_state_work_after_reset():
    ENV.close()
    reset_response = client.post("/reset", json={"task_id": DEFAULT_TASK_ID})
    assert reset_response.status_code == 200

    step_response = client.post("/step", json={"action": "r_up"})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert 0.0 <= step_payload["reward"]["value"] <= 1.0
    assert 0.0 <= step_payload["reward"]["accuracy_score"] <= 1.0
    assert 0.0 <= step_payload["reward"]["cost_efficiency"] <= 1.0
    assert 0.0 <= step_payload["reward"]["step_efficiency"] <= 1.0
    assert step_payload["info"]["step_count"] == 1
    assert step_payload["info"]["terminated_by"] == "in_progress"
    assert "observation" in step_payload

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["task_id"] == DEFAULT_TASK_ID
    assert state_payload["target_hz"] > 0.0
    assert state_payload["circuit_type"] in {"low_pass", "high_pass"}
    assert state_payload["step_count"] == 1
    assert 0.0 <= state_payload["current_normalized_error"] <= 1.0
    assert 0.0 <= state_payload["current_cost"] <= 1.0


def test_tasks_returns_deterministic_task_ids():
    ENV.close()
    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == {"task_ids": TASK_IDS}


def test_metadata_exposes_environment_summary():
    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "circuitrl"
    assert payload["description"]
    assert payload["version"] == "0.1.0"
    assert payload["task_ids"] == TASK_IDS
    assert payload["default_task_id"] == DEFAULT_TASK_ID


def test_schema_exposes_typed_models():
    response = client.get("/schema")

    assert response.status_code == 200
    payload = response.json()
    assert "properties" in payload["action"]
    assert "properties" in payload["observation"]
    assert "properties" in payload["reward"]
    assert "properties" in payload["step_info"]
    assert "properties" in payload["state"]
    assert "properties" in payload["task"]


def test_mcp_returns_json_rpc_payload():
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 1
    assert payload["result"]["serverInfo"]["name"] == "circuitrl"


def test_ui_catalog_exposes_default_task_and_order():
    response = client.get("/api/ui/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_task_id"] == DEFAULT_TASK_ID
    assert payload["task_ids"] == TASK_IDS
    assert payload["tasks"][0]["task_id"] == DEFAULT_TASK_ID
    assert payload["action_scale_factor"] > 1.0


def test_ui_preview_returns_single_frame_payload():
    response = client.get("/api/ui/preview", params={"task_id": DEFAULT_TASK_ID})

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["task_id"] == DEFAULT_TASK_ID
    assert len(payload["frames"]) == 1
    assert payload["frames"][0]["step"] == 0
    assert payload["summary"] is None
    assert payload["comparisons"] == []


def test_ui_episode_returns_full_playback_and_comparisons(monkeypatch):
    monkeypatch.setattr(
        "server.app.build_ui_episode_agent",
        lambda task_id: TabularValueIterationAgent({task_id: TASKS[task_id]}),
    )
    response = client.get("/api/ui/episode", params={"task_id": DEFAULT_TASK_ID})

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["task_id"] == DEFAULT_TASK_ID
    assert len(payload["frames"]) > 1
    assert payload["summary"]["steps_used"] >= 1
    assert payload["comparisons"][0]["baseline_name"] == "agent"
    assert {row["baseline_name"] for row in payload["comparisons"]} == {
        "agent",
        "random",
        "heuristic",
        "bruteforce",
    }
