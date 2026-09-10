from fastapi.testclient import TestClient

from api.main import app
from agent.classifier import classify
from agent.retriever import retrieve
from agent.escalation import escalation_decision
from agent.responder import generate_grounded_response


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "hiver-ai-support-agent"


def test_triage_endpoint():
    response = client.post(
        "/triage",
        json={"message": "Spotify keeps crashing when I open the app"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Spotify keeps crashing when I open the app"
    assert isinstance(data["intent"], str)
    assert isinstance(data["classifier_confidence"], float)
    assert isinstance(data["retrieval_similarity"], float)
    assert data["action"] in {"ANSWER", "CLARIFY", "ESCALATE"}
    assert isinstance(data["response"], str)


def test_classifier():
    result = classify("Spotify keeps crashing when I open the app")

    assert "intent" in result
    assert "confidence" in result
    assert isinstance(result["intent"], str)
    assert 0 <= result["confidence"] <= 1


def test_retriever():
    results = retrieve(
        "Spotify keeps crashing when I open the app",
        intent="APP_TECHNICAL",
        top_k=3
    )

    assert len(results) <= 3

    if results:
        assert "customer_case" in results[0]
        assert "support_response" in results[0]
        assert "similarity" in results[0]
        assert "combined_score" in results[0]


def test_security_escalation():
    result = escalation_decision(
        intent="ACCOUNT_SECURITY",
        classifier_confidence=0.8,
        retrieval_similarity=0.8
    )

    assert result["action"] == "ESCALATE"


def test_billing_escalation():
    result = escalation_decision(
        intent="BILLING_PAYMENT",
        classifier_confidence=0.8,
        retrieval_similarity=0.8
    )

    assert result["action"] == "ESCALATE"


def test_strong_retrieval_answers():
    result = escalation_decision(
        intent="APP_TECHNICAL",
        classifier_confidence=0.8,
        retrieval_similarity=0.4
    )

    assert result["action"] == "ANSWER"


def test_moderate_retrieval_clarifies():
    result = escalation_decision(
        intent="APP_TECHNICAL",
        classifier_confidence=0.8,
        retrieval_similarity=0.25
    )

    assert result["action"] == "CLARIFY"


def test_weak_evidence_clarifies():
    result = escalation_decision(
        intent="APP_TECHNICAL",
        classifier_confidence=0.1,
        retrieval_similarity=0.1
    )

    assert result["action"] == "CLARIFY"


def test_responder_without_cases():
    result = generate_grounded_response(
        query="Spotify is not working",
        intent="APP_TECHNICAL",
        retrieved_cases=[]
    )

    assert isinstance(result, dict)
    assert result["grounded"] is False
    assert result["needs_escalation"] is True
    assert isinstance(result["response"], str)