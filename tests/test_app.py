"""Smoke checks for the FastAPI/OpenEnv wrapper."""

from fastapi.testclient import TestClient

from server.app import DEFAULT_TASK_ID, ENV, TASK_IDS, app


client = TestClient(app)


def test_health_returns_ok():
    ENV.close()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
    assert 0.0 <= step_payload["reward"] <= 1.0
    assert "observation" in step_payload

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["task_id"] == DEFAULT_TASK_ID
    assert state_payload["step_count"] == 1


def test_tasks_returns_deterministic_task_ids():
    ENV.close()
    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == {"task_ids": TASK_IDS}
