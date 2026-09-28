from types import SimpleNamespace
from unittest.mock import patch

import pytest

from knowledge_agent.embeddings import provider as provider_module
from knowledge_agent.embeddings.provider import (
    DEFAULT_EMBEDDING_MODEL,
    DOCUMENT_TASK_TYPE,
    MAX_BATCH_SIZE,
    QUERY_TASK_TYPE,
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingQuotaError,
    GeminiEmbeddingProvider,
)


class FakeModels:
    def __init__(self, dimensions: int = 3, error: Exception | None = None, vectors=None):
        self.dimensions = dimensions
        self.error = error
        self.vectors = vectors
        self.calls: list[dict] = []
        self._issued = 0

    def embed_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": list(contents), "config": config})

        if self.error is not None:
            raise self.error

        if self.vectors is not None:
            values = self.vectors
        else:
            values = []
            for _ in contents:
                values.append([float(self._issued)] * self.dimensions)
                self._issued += 1

        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=list(v)) for v in values]
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


def build(**kwargs) -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider(client=FakeClient(), **kwargs)


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_missing_api_key_raises_config_error():
    with patch.object(provider_module, "GEMINI_API_KEY", None):
        with pytest.raises(EmbeddingConfigError, match="No API key"):
            GeminiEmbeddingProvider()


def test_api_key_is_taken_from_argument(monkeypatch):
    created: dict = {}

    def fake_client(**kwargs):
        created.update(kwargs)
        return FakeClient()

    with patch.object(provider_module, "GEMINI_API_KEY", None):
        with patch.object(provider_module.genai, "Client", fake_client):
            GeminiEmbeddingProvider(api_key="secret")

    assert created == {"api_key": "secret"}


def test_api_key_falls_back_to_environment():
    created: dict = {}

    def fake_client(**kwargs):
        created.update(kwargs)
        return FakeClient()

    with patch.object(provider_module, "GEMINI_API_KEY", "from-env"):
        with patch.object(provider_module.genai, "Client", fake_client):
            GeminiEmbeddingProvider()

    assert created == {"api_key": "from-env"}


def test_injected_client_needs_no_api_key():
    with patch.object(provider_module, "GEMINI_API_KEY", None):
        assert GeminiEmbeddingProvider(client=FakeClient()) is not None


def test_model_defaults_when_not_configured():
    with patch.object(provider_module, "GEMINI_EMBEDDING_MODEL", None):
        assert GeminiEmbeddingProvider(client=FakeClient()).model == DEFAULT_EMBEDDING_MODEL


def test_model_prefers_environment():
    with patch.object(provider_module, "GEMINI_EMBEDDING_MODEL", "env-model"):
        provider = GeminiEmbeddingProvider(client=FakeClient())

    assert provider.model == "env-model"


def test_explicit_model_wins():
    with patch.object(provider_module, "GEMINI_EMBEDDING_MODEL", "env-model"):
        provider = GeminiEmbeddingProvider(model="explicit", client=FakeClient())

    assert provider.model == "explicit"


def test_zero_batch_size_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="batch_size must be positive"):
        build(batch_size=0)


def test_negative_batch_size_is_rejected():
    with pytest.raises(EmbeddingConfigError):
        build(batch_size=-5)


def test_batch_size_above_maximum_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="exceeds the maximum"):
        build(batch_size=MAX_BATCH_SIZE + 1)


def test_non_positive_output_dimensionality_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="output_dimensionality"):
        build(output_dimensionality=0)


def test_config_error_is_an_embedding_error():
    with pytest.raises(EmbeddingError):
        build(batch_size=0)


def test_embed_documents_returns_one_vector_per_text():
    vectors = build().embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(isinstance(v, list) for v in vectors)
    assert all(len(v) == 3 for v in vectors)


def test_embed_documents_uses_document_task_type():
    provider = build()

    provider.embed_documents(["a"])

    assert provider._client.models.calls[0]["config"].task_type == DOCUMENT_TASK_TYPE


def test_embed_query_uses_query_task_type():
    provider = build()

    provider.embed_query("a question")

    assert provider._client.models.calls[0]["config"].task_type == QUERY_TASK_TYPE


def test_query_task_type_differs_from_document_task_type():
    assert DOCUMENT_TASK_TYPE != QUERY_TASK_TYPE


def test_embed_query_returns_single_vector():
    vector = build().embed_query("a question")

    assert isinstance(vector, list)
    assert all(isinstance(value, float) for value in vector)


def test_embed_documents_on_empty_input():
    assert build().embed_documents([]) == []


def test_embed_query_rejects_blank_text():
    with pytest.raises(EmbeddingConfigError, match="cannot be empty"):
        build().embed_query("   \n ")


def test_model_and_contents_are_passed_through():
    provider = build(model="my-model")

    provider.embed_documents(["x", "y"])

    call = provider._client.models.calls[0]
    assert call["model"] == "my-model"
    assert call["contents"] == ["x", "y"]


def test_output_dimensionality_is_passed_to_config():
    provider = build(output_dimensionality=256)

    provider.embed_documents(["a"])

    assert provider._client.models.calls[0]["config"].output_dimensionality == 256


def test_output_dimensionality_defaults_to_none():
    provider = build()

    provider.embed_documents(["a"])

    assert provider._client.models.calls[0]["config"].output_dimensionality is None


def test_inputs_are_split_into_batches():
    provider = build(batch_size=2)

    provider.embed_documents(["a", "b", "c", "d", "e"])

    calls = provider._client.models.calls
    assert [len(call["contents"]) for call in calls] == [2, 2, 1]


def test_batching_preserves_order():
    client = FakeClient(dimensions=1)
    provider = GeminiEmbeddingProvider(client=client, batch_size=2)

    vectors = provider.embed_documents(["a", "b", "c", "d", "e"])

    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_batch_size_one_sends_one_request_per_text():
    provider = build(batch_size=1)

    provider.embed_documents(["a", "b", "c"])

    assert len(provider._client.models.calls) == 3


def test_api_failure_is_wrapped():
    provider = GeminiEmbeddingProvider(client=FakeClient(error=RuntimeError("429 rate limit")))

    with pytest.raises(EmbeddingAPIError, match="rate limit"):
        provider.embed_documents(["a"])


def test_api_failure_preserves_cause():
    cause = RuntimeError("boom")
    provider = GeminiEmbeddingProvider(client=FakeClient(error=cause))

    with pytest.raises(EmbeddingAPIError) as info:
        provider.embed_documents(["a"])

    assert info.value.__cause__ is cause


def test_api_failure_during_query_is_wrapped():
    provider = GeminiEmbeddingProvider(client=FakeClient(error=RuntimeError("boom")))

    with pytest.raises(EmbeddingAPIError):
        provider.embed_query("question")


def test_api_error_is_an_embedding_error():
    provider = GeminiEmbeddingProvider(client=FakeClient(error=RuntimeError("boom")))

    with pytest.raises(EmbeddingError):
        provider.embed_documents(["a"])


def test_count_mismatch_raises():
    provider = GeminiEmbeddingProvider(
        client=FakeClient(vectors=[[1.0, 2.0]])
    )

    with pytest.raises(EmbeddingError, match="Expected 2 embeddings"):
        provider.embed_documents(["a", "b"])


def test_inconsistent_dimensions_raise():
    provider = GeminiEmbeddingProvider(
        client=FakeClient(vectors=[[1.0, 2.0], [1.0]])
    )

    with pytest.raises(EmbeddingError, match="Inconsistent embedding size"):
        provider.embed_documents(["a", "b"])


def test_empty_embedding_raises():
    provider = GeminiEmbeddingProvider(client=FakeClient(vectors=[[]]))

    with pytest.raises(EmbeddingError, match="empty embedding"):
        provider.embed_documents(["a"])


def test_response_without_embeddings_raises_embedding_error():
    client = FakeClient()
    client.models.embed_content = lambda **kwargs: SimpleNamespace(embeddings=None)
    provider = GeminiEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingError, match="Expected 1 embeddings"):
        provider.embed_documents(["a"])


def test_embedding_with_null_values_raises_embedding_error():
    client = FakeClient()
    client.models.embed_content = lambda **kwargs: SimpleNamespace(
        embeddings=[SimpleNamespace(values=None)]
    )
    provider = GeminiEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingError, match="empty embedding"):
        provider.embed_documents(["a"])


def test_sequences_other_than_lists_are_accepted():
    provider = build()

    assert len(provider.embed_documents(("a", "b"))) == 2


def test_documents_and_queries_share_one_client():
    provider = build()

    provider.embed_documents(["a"])
    provider.embed_query("b")

    assert len(provider._client.models.calls) == 2


class RateLimitError(Exception):
    def __init__(self, code: int, message: str = "quota"):
        super().__init__(message)
        self.code = code


class FlakyModels(FakeModels):
    def __init__(self, failures: int, code: int = 429, message: str = "quota"):
        super().__init__()
        self.remaining = failures
        self.code = code
        self.message = message

    def embed_content(self, model, contents, config=None):
        if self.remaining > 0:
            self.remaining -= 1
            raise RateLimitError(self.code, self.message)
        return super().embed_content(model, contents, config)


class RecordingSleep:
    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def build_flaky(failures: int, code: int = 429, message: str = "quota", **kwargs):
    client = FakeClient()
    client.models = FlakyModels(failures, code, message)
    sleep = RecordingSleep()
    kwargs.setdefault("backoff_seconds", 1.0)
    provider = GeminiEmbeddingProvider(client=client, sleep=sleep, **kwargs)
    return provider, sleep


def test_rate_limit_is_retried():
    provider, sleep = build_flaky(failures=2)

    vectors = provider.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert len(sleep.delays) == 2


def test_retry_stops_once_it_succeeds():
    provider, sleep = build_flaky(failures=1)

    provider.embed_documents(["a"])

    assert len(sleep.delays) == 1


def test_retry_gives_up_after_max_attempts():
    provider, sleep = build_flaky(failures=99, max_attempts=3)

    with pytest.raises(EmbeddingAPIError, match="after 3 attempt"):
        provider.embed_documents(["a"])

    assert len(sleep.delays) == 2


def test_non_retryable_error_is_not_retried():
    provider, sleep = build_flaky(failures=99, code=400)

    with pytest.raises(EmbeddingAPIError, match="after 1 attempt"):
        provider.embed_documents(["a"])

    assert sleep.delays == []


def test_backoff_grows_exponentially():
    provider, sleep = build_flaky(failures=99, backoff_seconds=2.0, max_attempts=4)

    with pytest.raises(EmbeddingAPIError):
        provider.embed_documents(["a"])

    assert sleep.delays == [2.0, 4.0, 8.0]


def test_server_retry_delay_is_honoured():
    provider, sleep = build_flaky(
        failures=1, message="quota exceeded, retryDelay: '51s'"
    )

    provider.embed_documents(["a"])

    assert sleep.delays == [51.0]


def test_invalid_max_attempts_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="max_attempts"):
        build(max_attempts=0)


def test_negative_backoff_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="backoff_seconds"):
        build(backoff_seconds=-1)


def test_single_attempt_disables_retry():
    provider, sleep = build_flaky(failures=99, max_attempts=1)

    with pytest.raises(EmbeddingAPIError, match="after 1 attempt"):
        provider.embed_documents(["a"])

    assert sleep.delays == []


def test_daily_quota_exhaustion_is_not_retried():
    message = "quota exceeded for metric: embed_content_free_tier_requests PerDay, limit 1000"
    provider, sleep = build_flaky(failures=99, message=message)

    with pytest.raises(EmbeddingQuotaError, match="Daily embedding quota"):
        provider.embed_documents(["a"])

    assert sleep.delays == []


def test_daily_quota_error_names_the_model():
    provider, _ = build_flaky(failures=99, message="PerDay limit 1000")

    with pytest.raises(EmbeddingQuotaError, match="gemini"):
        provider.embed_documents(["a"])


def test_daily_quota_error_is_an_api_error():
    provider, _ = build_flaky(failures=99, message="PerDay limit 1000")

    with pytest.raises(EmbeddingAPIError):
        provider.embed_documents(["a"])


def test_per_minute_limit_is_still_retried():
    message = "quota exceeded EmbedContentRequestsPerMinute limit 100, retryDelay '5s'"
    provider, sleep = build_flaky(failures=1, message=message)

    provider.embed_documents(["a"])

    assert sleep.delays == [5.0]
