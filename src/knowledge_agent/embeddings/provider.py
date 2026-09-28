import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from itertools import batched

from google import genai

from knowledge_agent.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
MAX_BATCH_SIZE = 100
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRY_DELAY_PATTERN = re.compile(r"retryDelay\D{0,4}(\d+(?:\.\d+)?)")
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


class EmbeddingError(Exception):
    pass


class EmbeddingConfigError(EmbeddingError):
    pass


class EmbeddingAPIError(EmbeddingError):
    pass


class EmbeddingQuotaError(EmbeddingAPIError):
    pass


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = MAX_BATCH_SIZE,
        output_dimensionality: int | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        client: genai.Client | None = None,
    ) -> None:
        if batch_size <= 0:
            raise EmbeddingConfigError(f"batch_size must be positive, got {batch_size}")

        if batch_size > MAX_BATCH_SIZE:
            raise EmbeddingConfigError(
                f"batch_size ({batch_size}) exceeds the maximum of {MAX_BATCH_SIZE}"
            )

        if output_dimensionality is not None and output_dimensionality <= 0:
            raise EmbeddingConfigError(
                f"output_dimensionality must be positive, got {output_dimensionality}"
            )

        if max_attempts <= 0:
            raise EmbeddingConfigError(f"max_attempts must be positive, got {max_attempts}")

        if backoff_seconds < 0:
            raise EmbeddingConfigError(
                f"backoff_seconds cannot be negative, got {backoff_seconds}"
            )

        self.model = model or GEMINI_EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODEL
        self.batch_size = batch_size
        self.output_dimensionality = output_dimensionality
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or GEMINI_API_KEY

        if not resolved_key:
            raise EmbeddingConfigError(
                "No API key: set GEMINI_API_KEY in the environment or pass api_key=..."
            )

        self._client = genai.Client(api_key=resolved_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts), DOCUMENT_TASK_TYPE)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingConfigError("Query text cannot be empty")

        return self._embed([text], QUERY_TASK_TYPE)[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []

        for batch in batched(texts, self.batch_size):
            vectors.extend(self._request(list(batch), task_type))

        return vectors

    def _request(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=self._config(task_type),
                )
                break
            except Exception as exc:
                if attempt == self.max_attempts or not self._is_retryable(exc):
                    raise self._wrap(exc, attempt) from exc

                self._sleep(self._delay(attempt, exc))

        embeddings = response.embeddings or []
        vectors = [
            list(embedding.values) if embedding.values else [] for embedding in embeddings
        ]

        self._validate(texts, vectors)

        return vectors

    def _wrap(self, exc: Exception, attempts: int) -> EmbeddingAPIError:
        if self._is_quota_exhausted(exc):
            return EmbeddingQuotaError(
                f"Daily embedding quota for '{self.model}' is exhausted: {exc}. "
                "Wait for the quota to reset, reduce the corpus, or use a paid API key."
            )

        return EmbeddingAPIError(
            f"Embedding request to '{self.model}' failed after {attempts} attempt(s): {exc}"
        )

    @staticmethod
    def _is_quota_exhausted(exc: Exception) -> bool:
        message = str(exc).lower()
        return "perday" in message or "per day" in message

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if GeminiEmbeddingProvider._is_quota_exhausted(exc):
            return False

        code = getattr(exc, "code", None)
        if code is None:
            code = getattr(exc, "status", None)
        if isinstance(code, int):
            return code in RETRYABLE_STATUS_CODES
        return False

    def _delay(self, attempt: int, exc: Exception) -> float:
        match = RETRY_DELAY_PATTERN.search(str(exc))

        if match:
            return float(match.group(1))

        return self.backoff_seconds * (2 ** (attempt - 1))

    def _config(self, task_type: str) -> genai.types.EmbedContentConfig:
        return genai.types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.output_dimensionality,
        )

    def _validate(self, texts: list[str], vectors: list[list[float]]) -> None:
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings from '{self.model}', got {len(vectors)}"
            )

        if not vectors:
            return

        expected = len(vectors[0])

        if expected == 0:
            raise EmbeddingError(f"'{self.model}' returned an empty embedding")

        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    f"Inconsistent embedding size from '{self.model}': "
                    f"expected {expected}, got {len(vector)}"
                )
