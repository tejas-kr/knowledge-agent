from collections.abc import Sequence
from itertools import batched

import httpx

from knowledge_agent.config import (
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
)
from knowledge_agent.embeddings.provider import (
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingProvider,
    validate_vectors,
)
from knowledge_agent.embeddings.space import NOMIC_SPACE, EmbeddingSpace, space_for

DEFAULT_OLLAMA_BATCH_SIZE = 64
DEFAULT_OLLAMA_TIMEOUT = 600.0
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
EMBED_PATH = "/api/embed"
SECONDS_PER_MINUTE = 60


def _resolve_timeout(timeout: float | None) -> float:
    raw = timeout if timeout is not None else OLLAMA_TIMEOUT

    if raw is None or raw == "":
        return DEFAULT_OLLAMA_TIMEOUT

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise EmbeddingConfigError(f"timeout must be a number, got {raw!r}") from exc

    if value <= 0:
        raise EmbeddingConfigError(f"timeout must be positive, got {value}")

    return value


def _pick_space(
    requested: EmbeddingSpace | None,
    model: str,
    dimensions: int,
) -> EmbeddingSpace:
    if requested is not None:
        return requested

    if model == NOMIC_SPACE.model and dimensions == NOMIC_SPACE.dimensions:
        return NOMIC_SPACE

    return space_for(model, dimensions, "ollama")


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        batch_size: int = DEFAULT_OLLAMA_BATCH_SIZE,
        timeout: float | None = None,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
        space: EmbeddingSpace | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if batch_size <= 0:
            raise EmbeddingConfigError(f"batch_size must be positive, got {batch_size}")

        resolved_timeout = _resolve_timeout(timeout)

        if not keep_alive.strip():
            raise EmbeddingConfigError("keep_alive cannot be empty")

        resolved_host = (host or OLLAMA_HOST or "").strip()

        if not resolved_host:
            raise EmbeddingConfigError(
                "No Ollama host: set OLLAMA_HOST or pass host=..."
            )

        resolved_space = space or NOMIC_SPACE
        resolved_model = model or OLLAMA_EMBEDDING_MODEL or resolved_space.model

        if not resolved_model.strip():
            raise EmbeddingConfigError("model cannot be empty")

        self.model = resolved_model
        self.host = resolved_host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = resolved_timeout
        self.keep_alive = keep_alive
        self._space = _pick_space(space, resolved_model, resolved_space.dimensions)
        self._client = (
            client if client is not None else httpx.Client(timeout=resolved_timeout)
        )

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefix = self._space.document_instruction
        return self._embed([f"{prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingConfigError("Query text cannot be empty")

        return self._embed([f"{self._space.query_instruction}{text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []

        for batch in batched(texts, self.batch_size):
            vectors.extend(self._request(list(batch)))

        return vectors

    def _request(self, texts: list[str]) -> list[list[float]]:
        body = {
            "model": self.model,
            "input": texts,
            "truncate": True,
            "keep_alive": self.keep_alive,
        }

        try:
            response = self._client.post(f"{self.host}{EMBED_PATH}", json=body)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise EmbeddingAPIError(
                f"Ollama embedding request to '{self.model}' at "
                f"'{self.host}' failed: {exc}"
            ) from exc

        vectors = [list(vector) for vector in (payload.get("embeddings") or [])]

        validate_vectors(texts, vectors, self.model)
        self._verify_width(vectors)

        return vectors

    def _verify_width(self, vectors: list[list[float]]) -> None:
        """Guard the space's declared width against what the model returned.

        A silent width change would be caught later by the store, but only
        after the vectors were already written into the collection.
        """
        for vector in vectors:
            if len(vector) == self._space.dimensions:
                continue

            raise EmbeddingError(
                f"'{self.model}' returned {len(vector)}-dim vectors but space "
                f"'{self._space.key}' is registered as {self._space.dimensions}-dim. "
                f"Its collection is bound to the old width."
            )