from fastapi.testclient import TestClient
from ai_service.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/ai/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "retrieval" in data
    assert "evaluator" in data

def test_ask_endpoint_hospitalization():
    payload = {
        "policy_id": "policy_health_01",
        "question": "Is hospitalization covered under this policy?"
    }
    response = client.post("/ai/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "COVERED"
    assert data["evaluator_status"] == "VERIFIED_PASS"
    assert len(data["citations"]) > 0
    assert data["confidence_score"] >= 0.70

def test_ask_endpoint_cosmetic_exclusion():
    payload = {
        "policy_id": "policy_health_01",
        "question": "Can I claim for cosmetic surgery?"
    }
    response = client.post("/ai/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "NOT COVERED"
    assert data["evaluator_status"] == "VERIFIED_PASS"
