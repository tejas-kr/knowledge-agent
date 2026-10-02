"""HTTP surface over the answer pipeline.

Holds one store per space for the process lifetime. Chroma keeps file handles
open on Windows, and rebuilding a store per request would pay that cost on
every call; the API only ever reads, so a long-lived handle is safe.
"""

from collections.abc import Callable
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from knowledge_agent.config import RAG_TOP_K
from knowledge_agent.embeddings.resolver import resolve_provider
from knowledge_agent.embeddings.space import EmbeddingSpace
from knowledge_agent.rag.generator import GenerationError, Generator, build_generator
from knowledge_agent.rag.pipeline import RAGError, answer_question
from knowledge_agent.retrieval.vector_store import ChromaVectorStore, VectorStore, VectorStoreError
from knowledge_agent.schemas.rag import AskRequest, RAGAnswer

DEFAULT_SPACE = "nomic-768"


def _open_store(space_key: str) -> tuple[ChromaVectorStore, EmbeddingSpace]:
    resolution = resolve_provider(space_key)
    store = ChromaVectorStore(
        provider=resolution.provider, space=resolution.space
    )
    return store, resolution.space


def create_app(
    store_opener: Callable[[str], tuple[VectorStore, EmbeddingSpace]] | None = None,
    generator_factory: Callable[[], Generator] | None = None,
) -> FastAPI:
    """Build the app.

    Both collaborators are injectable so the tests can run without a network
    call, a real API key, or a real Chroma database.
    """
    open_store = store_opener or _cached_store
    make_generator = generator_factory or _cached_generator

    app = FastAPI(title="knowledge-agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=RAGAnswer)
    def ask(request: AskRequest) -> RAGAnswer:
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="question cannot be empty")

        try:
            store, space = open_store(request.space or DEFAULT_SPACE)
            generator = make_generator()
        except (GenerationError, VectorStoreError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            return answer_question(
                request.question,
                store,
                generator,
                space=space,
                k=request.k or RAG_TOP_K,
                min_distance=request.min_distance,
            )
        except RAGError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


@lru_cache(maxsize=None)
def _cached_store(space_key: str) -> tuple[ChromaVectorStore, EmbeddingSpace]:
    return _open_store(space_key)


@lru_cache(maxsize=1)
def _cached_generator() -> Generator:
    """Build the generator once.

    It holds an HTTP client and, for Gemini, a connection pool, so rebuilding
    it per request would pay that cost on every call. It is still lazy:
    importing this module must not require an API key or a running Ollama.
    """
    return build_generator()


app = create_app()
