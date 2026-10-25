"""Tests des endpoints HTTP."""
from fastapi.testclient import TestClient

from tests.conftest import FakeQueryEngine


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client: TestClient, fake_query_engine: FakeQueryEngine) -> None:
    response = client.post("/query", json={"question": "Combien coûte Premium ?"})

    assert response.status_code == 200
    assert fake_query_engine.received_questions == ["Combien coûte Premium ?"]
    assert response.json() == {
        "answer": "Réponse factice",
        "sources": [
            {"file_name": "tarifs.md", "score": 0.91, "text": "Le plan Premium coûte 10 €."},
            {"file_name": "?", "score": None, "text": "Le plan Gratuit est limité."},
        ],
    }


def test_query_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_rejects_out_of_range_top_k(client: TestClient) -> None:
    assert client.post("/query", json={"question": "ok", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "ok", "top_k": 11}).status_code == 422


def test_query_hides_generation_error_details(client: TestClient, fake_query_engine: FakeQueryEngine) -> None:
    fake_query_engine.error = RuntimeError("détail interne à ne pas exposer")

    response = client.post("/query", json={"question": "Question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Erreur du service de génération"}
