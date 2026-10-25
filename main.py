"""API FastAPI exposant le pipeline RAG Plannea."""
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

import rag


class QueryRequest(BaseModel):
    """Corps de requête de /query et /retrieve."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=10, description="Nombre de chunks récupérés")


class SourceChunk(BaseModel):
    """Chunk de contexte utilisé pour produire la réponse."""

    file_name: str
    score: Optional[float]
    text: str


class QueryResponse(BaseModel):
    """Réponse de /query."""

    answer: str
    sources: list[SourceChunk]


class RetrieveResponse(BaseModel):
    """Réponse de /retrieve : le prompt rempli, prêt à être envoyé à n'importe quel LLM."""

    prompt: str
    sources: list[SourceChunk]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge les modèles et connecte l'index Supabase une seule fois au démarrage."""
    load_dotenv()
    mistral_api_key = rag.read_optional_environment_variable("MISTRAL_API_KEY")
    rag.configure_models(mistral_api_key)
    app.state.generation_enabled = mistral_api_key is not None
    app.state.index = rag.load_index(rag.read_required_environment_variable("SUPABASE_DB_URL"))
    yield


app = FastAPI(title="Plannea RAG API", version="1.0.0", lifespan=lifespan)


def build_source_chunks(nodes: list[NodeWithScore]) -> list[SourceChunk]:
    """Convertit des nœuds LlamaIndex en `SourceChunk` sérialisables."""
    return [
        SourceChunk(
            file_name=source.node.metadata.get("file_name", "?"),
            score=source.score,
            text=source.node.text.strip(),
        )
        for source in nodes
    ]


def build_query_response(llama_response: Any) -> QueryResponse:
    """Convertit une réponse LlamaIndex en `QueryResponse` sérialisable."""
    return QueryResponse(answer=str(llama_response).strip(), sources=build_source_chunks(llama_response.source_nodes))


@app.get("/health")
def health() -> dict[str, str]:
    """Sonde de disponibilité."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Interroge le RAG. Endpoint synchrone : FastAPI l'exécute dans un thread pool."""
    if not request.app.state.generation_enabled:
        raise HTTPException(status_code=503, detail="Génération désactivée : MISTRAL_API_KEY non configurée")
    query_engine = rag.create_query_engine(request.app.state.index, payload.top_k)
    try:
        llama_response = query_engine.query(payload.question)
    except Exception as error:  # le contenu de la question n'est volontairement pas loggé (RGPD)
        raise HTTPException(status_code=502, detail="Erreur du service de génération") from error
    return build_query_response(llama_response)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: QueryRequest, request: Request) -> RetrieveResponse:
    """Partie retrieval seule : renvoie le prompt rempli et ses sources, sans appel au LLM ni clé Mistral."""
    try:
        nodes = rag.retrieve_nodes(request.app.state.index, payload.question, payload.top_k)
    except Exception as error:  # le contenu de la question n'est volontairement pas loggé (RGPD)
        raise HTTPException(status_code=502, detail="Erreur du service de recherche") from error
    prompt = rag.build_prompt(rag.build_context(nodes), payload.question)
    return RetrieveResponse(prompt=prompt, sources=build_source_chunks(nodes))
