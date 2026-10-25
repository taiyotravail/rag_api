"""Tests unitaires des fonctions de `rag`."""
from typing import Iterator

import pytest
from llama_index.core import MockEmbedding, Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.mistralai import MistralAI

import rag
from tests.conftest import make_source_nodes


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


def test_build_context_includes_metadata_and_skips_empty_chunks() -> None:
    nodes = make_source_nodes() + [NodeWithScore(node=TextNode(text="   "), score=0.1)]

    assert rag.build_context(nodes) == (
        "file_name: tarifs.md\n\n  Le plan Premium coûte 10 €.\n\nLe plan Gratuit est limité."
    )


def test_build_context_of_no_nodes_is_empty() -> None:
    assert rag.build_context([]) == ""


def test_build_prompt_fills_both_template_fields() -> None:
    prompt = rag.build_prompt("CONTEXTE", "QUESTION ?")

    assert "{context_str}" not in prompt and "{query_str}" not in prompt
    assert "---------------------\nCONTEXTE\n---------------------" in prompt
    assert prompt.endswith("Question : QUESTION ?\nRéponse : ")


def test_build_prompt_keeps_braces_from_user_input_literal() -> None:
    assert "Question : {context_str} {0}\n" in rag.build_prompt("ctx", "{context_str} {0}")


def test_retrieve_prompt_matches_prompt_sent_by_query_engine() -> None:
    """Garantit que /retrieve produit exactement le prompt que /query envoie au LLM (MockLLM renvoie son prompt)."""
    nodes = make_source_nodes()
    synthesizer = get_response_synthesizer(
        llm=MockLLM(), response_mode="compact", text_qa_template=PromptTemplate(rag.QA_TEMPLATE_STR)
    )

    prompt_sent_to_llm = str(synthesizer.synthesize("Combien ?", nodes))

    assert rag.build_prompt(rag.build_context(nodes), "Combien ?") == prompt_sent_to_llm


def test_retrieve_nodes_returns_top_k_nodes_from_index() -> None:
    nodes = [TextNode(text=f"Chunk {number}") for number in range(5)]
    index = VectorStoreIndex(nodes, embed_model=MockEmbedding(embed_dim=rag.EMBEDDING_DIMENSION))

    retrieved = rag.retrieve_nodes(index, "Question", top_k=2)

    assert len(retrieved) == 2
    assert all(isinstance(node, NodeWithScore) for node in retrieved)
