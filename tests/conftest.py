"""Fixtures partagées : l'API est testée sans modèle d'embedding, sans Supabase et sans appel Mistral."""
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

import main
import rag


class FakeLlamaResponse:
    """Imite une réponse LlamaIndex : texte généré + nœuds sources."""

    def __init__(self, answer: str, source_nodes: list[NodeWithScore]) -> None:
        self.answer = answer
        self.source_nodes = source_nodes

    def __str__(self) -> str:
        return self.answer


class FakeQueryEngine:
    """Query Engine factice qui renvoie une réponse prédéfinie ou lève l'erreur configurée."""

    def __init__(self, response: FakeLlamaResponse, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.received_questions: list[str] = []

    def query(self, question: str) -> FakeLlamaResponse:
        """Enregistre la question reçue puis renvoie la réponse (ou lève l'erreur)."""
        self.received_questions.append(question)
        if self.error is not None:
            raise self.error
        return self.response


class FakeRetriever:
    """Remplace `rag.retrieve_nodes` : renvoie des nœuds prédéfinis ou lève l'erreur configurée."""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes
        self.error: Exception | None = None
        self.calls: list[tuple[object, str, int]] = []

    def __call__(self, index: object, question: str, top_k: int) -> list[NodeWithScore]:
        """Enregistre l'appel puis renvoie les nœuds (ou lève l'erreur)."""
        self.calls.append((index, question, top_k))
        if self.error is not None:
            raise self.error
        return self.nodes


def make_source_nodes() -> list[NodeWithScore]:
    """Construit deux chunks sources fictifs (aucune donnée personnelle)."""
    return [
        NodeWithScore(node=TextNode(text="  Le plan Premium coûte 10 €.  ", metadata={"file_name": "tarifs.md"}), score=0.91),
        NodeWithScore(node=TextNode(text="Le plan Gratuit est limité.", metadata={}), score=None),
    ]


@pytest.fixture
def source_nodes() -> list[NodeWithScore]:
    """Chunks sources fictifs."""
    return make_source_nodes()


@pytest.fixture
def fake_query_engine(source_nodes: list[NodeWithScore]) -> FakeQueryEngine:
    """Query Engine factice répondant « Réponse factice »."""
    return FakeQueryEngine(FakeLlamaResponse("  Réponse factice  ", source_nodes))


@pytest.fixture
def fake_retriever(source_nodes: list[NodeWithScore]) -> FakeRetriever:
    """Retriever factice renvoyant les chunks sources fictifs."""
    return FakeRetriever(source_nodes)


@pytest.fixture
def fake_index() -> object:
    """Index factice : seul son identité compte (il est transmis tel quel au retriever)."""
    return object()


@pytest.fixture
def configured_mistral_keys(
    monkeypatch: pytest.MonkeyPatch,
    fake_query_engine: FakeQueryEngine,
    fake_retriever: FakeRetriever,
    fake_index: object,
) -> list[str | None]:
    """Neutralise le `.env` local, HuggingFace, Supabase et Mistral ; enregistre les clés passées à `configure_models`."""
    keys: list[str | None] = []
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://fake")
    monkeypatch.setattr(rag, "configure_models", keys.append)
    monkeypatch.setattr(rag, "load_index", lambda url: fake_index)
    monkeypatch.setattr(rag, "create_query_engine", lambda index, top_k: fake_query_engine)
    monkeypatch.setattr(rag, "retrieve_nodes", fake_retriever)
    return keys


@pytest.fixture
def client(configured_mistral_keys: list[str | None], monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client HTTP avec une clé Mistral configurée (génération activée)."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_mistral_key(
    configured_mistral_keys: list[str | None], monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Client HTTP sans clé Mistral (génération désactivée)."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with TestClient(main.app) as test_client:
        yield test_client
