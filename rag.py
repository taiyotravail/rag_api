"""Pipeline RAG (LlamaIndex + Mistral + Supabase) isolé de la couche HTTP."""
import os
from typing import Any, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.vector_stores.supabase import SupabaseVectorStore

LLM_MODEL_NAME = "ministral-8b-2410"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384
COLLECTION_NAME = "documents"

QA_TEMPLATE_STR = (
    "Tu es un assistant factuel.\n"
    "Réponds en te basant uniquement sur ce contexte :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Si la réponse n'est pas dans le contexte, indique que tu ne sais pas.\n"
    "Question : {query_str}\n"
    "Réponse : "
)


def configure_models(mistral_api_key: Optional[str]) -> None:
    """Initialise le LLM Mistral et le modèle d'embedding dans les Settings globaux de LlamaIndex.

    Sans clé, le LLM est explicitement désactivé : LlamaIndex ne se rabat pas silencieusement sur OpenAI.
    """
    Settings.llm = MistralAI(model=LLM_MODEL_NAME, api_key=mistral_api_key) if mistral_api_key else None
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        text_instruction="passage: ",
        query_instruction="query: ",
    )


def load_index(postgres_connection_string: str) -> VectorStoreIndex:
    """Reconnecte l'index au vector store Supabase existant, sans recalculer d'embeddings."""
    vector_store = SupabaseVectorStore(
        postgres_connection_string=postgres_connection_string,
        dimension=EMBEDDING_DIMENSION,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)


def create_query_engine(index: VectorStoreIndex, top_k: int) -> Any:
    """Construit un Query Engine avec le prompt factuel et le nombre de chunks demandé."""
    return index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=PromptTemplate(QA_TEMPLATE_STR),
    )


def retrieve_nodes(index: VectorStoreIndex, question: str, top_k: int) -> list[NodeWithScore]:
    """Récupère les `top_k` chunks les plus proches de la question, sans aucun appel au LLM."""
    return index.as_retriever(similarity_top_k=top_k).retrieve(question)


def build_context(nodes: list[NodeWithScore]) -> str:
    """Assemble le `context_str` comme le synthétiseur `compact` de LlamaIndex utilisé par `/query`.

    Métadonnées visibles par le LLM incluses, chunks vides ignorés, séparateur : une ligne vide.
    """
    chunks = (node.node.get_content(metadata_mode=MetadataMode.LLM).strip() for node in nodes)
    return "\n\n".join(chunk for chunk in chunks if chunk)


def build_prompt(context: str, question: str) -> str:
    """Remplit `QA_TEMPLATE_STR` avec le contexte et la question : le prompt prêt à envoyer à un LLM."""
    return PromptTemplate(QA_TEMPLATE_STR).format(context_str=context, query_str=question)


def read_required_environment_variable(name: str) -> str:
    """Retourne la variable d'environnement `name` ou lève une ValueError si elle est absente."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Variable {name} introuvable dans l'environnement / .env")
    return value


def read_optional_environment_variable(name: str) -> Optional[str]:
    """Retourne la variable d'environnement `name`, ou None si elle est absente ou vide."""
    return os.environ.get(name) or None
