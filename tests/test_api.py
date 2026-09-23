from fastapi.testclient import TestClient

from app.api import app, get_bundle


def test_health_and_prediction(monkeypatch, demo_model_path):
    monkeypatch.setenv("FERTILITY_MODEL_PATH", str(demo_model_path))
    get_bundle.cache_clear()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["demo_only"] is True

    response = client.post(
        "/v1/predict",
        json={
            "current_age": 31,
            "residence": 2,
            "education": 2,
            "wealth": 3,
            "in_union": 1,
        },
    )
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


def test_api_rejects_invalid_age(monkeypatch, demo_model_path):
    monkeypatch.setenv("FERTILITY_MODEL_PATH", str(demo_model_path))
    get_bundle.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/v1/predict",
        json={
            "current_age": 52,
            "residence": 2,
            "education": 2,
            "wealth": 3,
            "in_union": 1,
        },
    )
    assert response.status_code == 422


def test_root_and_feature_contract():
    client = TestClient(app)
    assert client.get("/").json()["documentation"] == "/docs"

    response = client.get("/v1/features")
    assert response.status_code == 200
    assert response.json()["feature_order"] == [
        "current_age",
        "residence",
        "education",
        "wealth",
        "in_union",
    ]
    assert response.json()["max_batch_size"] == 500


def test_batch_prediction_and_size_validation(monkeypatch, demo_model_path):
    monkeypatch.setenv("FERTILITY_MODEL_PATH", str(demo_model_path))
    get_bundle.cache_clear()
    client = TestClient(app)
    valid_record = {
        "current_age": 31,
        "residence": 2,
        "education": 2,
        "wealth": 3,
        "in_union": 1,
    }

    response = client.post("/v1/predict/batch", json={"records": [valid_record, valid_record]})
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2

    empty_response = client.post("/v1/predict/batch", json={"records": []})
    assert empty_response.status_code == 422


def test_agent_endpoint(monkeypatch, demo_model_path):
    monkeypatch.setenv("FERTILITY_MODEL_PATH", str(demo_model_path))
    get_bundle.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/v1/agent/run",
        json={
            "intent": "assess_risk",
            "record": {
                "current_age": 31,
                "residence": 2,
                "education": 2,
                "wealth": 3,
                "in_union": 1,
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["tool_calls"][0]["tool"] == "predict_risk"
    assert response.json()["tool_calls"][-1]["tool"] == "generate_report"

    invalid_response = client.post("/v1/agent/run", json={"intent": "assess_risk"})
    assert invalid_response.status_code == 422


def test_agent_chat_plans_and_executes(monkeypatch, demo_model_path):
    monkeypatch.setenv("FERTILITY_MODEL_PATH", str(demo_model_path))
    get_bundle.cache_clear()
    client = TestClient(app)
    record = {
        "current_age": 31,
        "residence": 2,
        "education": 2,
        "wealth": 3,
        "in_union": 1,
    }
    response = client.post(
        "/v1/agent/chat",
        json={"message": "Is profile ka risk score explain karo", "record": record},
    )
    assert response.status_code == 200
    assert response.json()["plan"]["intent"] == "assess_risk"
    assert response.json()["response"]["tool_calls"][0]["tool"] == "predict_risk"

    missing_context = client.post(
        "/v1/agent/chat", json={"message": "Compare both scenarios"}
    )
    assert missing_context.status_code == 422
