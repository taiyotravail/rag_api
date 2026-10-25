"""Tests unitaires des fonctions de `rag`."""
from typing import Iterator

import pytest
from llama_index.core import MockEmbedding, Settings
from llama_index.core.llms.mock import MockLLM
from llama_index.llms.mistralai import MistralAI

import rag


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remplace HuggingFace par un faux embedding et restaure les Settings globaux après le test."""
    previous_llm, previous_embed_model = Settings._llm, Settings._embed_model
    monkeypatch.setattr(rag, "HuggingFaceEmbedding", lambda **kwargs: MockEmbedding(embed_dim=rag.EMBEDDING_DIMENSION))
    yield
    Settings._llm, Settings._embed_model = previous_llm, previous_embed_model


def test_configure_models_with_key_uses_mistral(isolated_settings: None) -> None:
    rag.configure_models("fake-key")

    assert isinstance(Settings.llm, MistralAI)
    assert Settings.llm.model == rag.LLM_MODEL_NAME


def test_configure_models_without_key_disables_llm(isolated_settings: None) -> None:
    rag.configure_models(None)

    assert isinstance(Settings.llm, MockLLM)


@pytest.mark.parametrize("value, expected", [("value", "value"), ("", None), (None, None)])
def test_read_optional_environment_variable(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("SOME_VARIABLE", raising=False)
    else:
        monkeypatch.setenv("SOME_VARIABLE", value)

    assert rag.read_optional_environment_variable("SOME_VARIABLE") == expected


def test_read_required_environment_variable_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_VARIABLE", "value")

    assert rag.read_required_environment_variable("SOME_VARIABLE") == "value"


@pytest.mark.parametrize("value", [None, ""])
def test_read_required_environment_variable_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("SOME_VARIABLE", raising=False)
    else:
        monkeypatch.setenv("SOME_VARIABLE", value)

    with pytest.raises(ValueError, match="SOME_VARIABLE"):
        rag.read_required_environment_variable("SOME_VARIABLE")
