"""Tests des endpoints HTTP."""
from fastapi.testclient import TestClient

import rag
from tests.conftest import FakeQueryEngine, FakeRetriever

EXPECTED_SOURCES = [
    {"file_name": "tarifs.md", "score": 0.91, "text": "Le plan Premium coûte 10 €."},
    {"file_name": "?", "score": None, "text": "Le plan Gratuit est limité."},
]


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client: TestClient, fake_query_engine: FakeQueryEngine) -> None:
    response = client.post("/query", json={"question": "Combien coûte Premium ?"})

    assert response.status_code == 200
    assert fake_query_engine.received_questions == ["Combien coûte Premium ?"]
    assert response.json() == {"answer": "Réponse factice", "sources": EXPECTED_SOURCES}


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


def test_startup_passes_mistral_key_to_models(client: TestClient, configured_mistral_keys: list[str | None]) -> None:
    assert configured_mistral_keys == ["fake-key"]


def test_startup_without_mistral_key_disables_llm(
    client_without_mistral_key: TestClient, configured_mistral_keys: list[str | None]
) -> None:
    assert configured_mistral_keys == [None]
    assert client_without_mistral_key.get("/health").status_code == 200


def test_query_without_mistral_key_returns_503(
    client_without_mistral_key: TestClient, fake_query_engine: FakeQueryEngine
) -> None:
    response = client_without_mistral_key.post("/query", json={"question": "Question"})

    assert response.status_code == 503
    assert "MISTRAL_API_KEY" in response.json()["detail"]
    assert fake_query_engine.received_questions == []


def test_retrieve_returns_filled_prompt_and_sources(
    client: TestClient, fake_retriever: FakeRetriever, fake_index: object
) -> None:
    response = client.post("/retrieve", json={"question": "Combien coûte Premium ?", "top_k": 5})

    assert response.status_code == 200
    assert fake_retriever.calls == [(fake_index, "Combien coûte Premium ?", 5)]
    expected_prompt = rag.build_prompt(rag.build_context(fake_retriever.nodes), "Combien coûte Premium ?")
    assert response.json() == {"prompt": expected_prompt, "sources": EXPECTED_SOURCES}


def test_retrieve_works_without_mistral_key(client_without_mistral_key: TestClient) -> None:
    response = client_without_mistral_key.post("/retrieve", json={"question": "Question"})

    assert response.status_code == 200
    assert "Question : Question" in response.json()["prompt"]


def test_retrieve_uses_default_top_k(client: TestClient, fake_retriever: FakeRetriever) -> None:
    client.post("/retrieve", json={"question": "Question"})

    assert fake_retriever.calls[0][2] == 3


def test_retrieve_rejects_invalid_payload(client: TestClient) -> None:
    assert client.post("/retrieve", json={"question": ""}).status_code == 422
    assert client.post("/retrieve", json={"question": "ok", "top_k": 11}).status_code == 422


def test_retrieve_hides_search_error_details(client: TestClient, fake_retriever: FakeRetriever) -> None:
    fake_retriever.error = RuntimeError("détail interne à ne pas exposer")

    response = client.post("/retrieve", json={"question": "Question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Erreur du service de recherche"}
