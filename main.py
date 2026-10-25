"""API FastAPI exposant le pipeline RAG Plannea."""
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import rag


class QueryRequest(BaseModel):
    """Corps de requête de /query."""

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


def build_query_response(llama_response: Any) -> QueryResponse:
    """Convertit une réponse LlamaIndex en `QueryResponse` sérialisable."""
    sources = [
        SourceChunk(
            file_name=source.node.metadata.get("file_name", "?"),
            score=source.score,
            text=source.node.text.strip(),
        )
        for source in llama_response.source_nodes
    ]
    return QueryResponse(answer=str(llama_response).strip(), sources=sources)


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
