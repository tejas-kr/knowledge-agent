import pytest

from knowledge_agent.embeddings.ollama import (
    DEFAULT_OLLAMA_BATCH_SIZE,
    OllamaEmbeddingProvider,
)
from knowledge_agent.embeddings.provider import (
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingError,
)
from knowledge_agent.embeddings.space import (
    DOCUMENT_INSTRUCTION,
    NOMIC_DIMENSIONS,
    NOMIC_SPACE,
    QUERY_INSTRUCTION,
)

HOST = "http://127.0.0.1:11434"


def vec(first: float) -> list[float]:
    return [first] + [0.0] * (NOMIC_DIMENSIONS - 1)


class FakeResponse:
    def __init__(self, payload: dict | None = None, status: int = 200):
        self._payload = payload if payload is not None else {"embeddings": []}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, responses=None):
        self.calls: list[dict] = []
        self.responses = responses or []

    def post(self, url, json=None, **kwargs):
        self.calls.append({"url": url, "json": json})

        if self.responses:
            return self.responses.pop(0)

        return FakeResponse({"embeddings": [vec(1.0) for _ in json["input"]]})


def build(client=None, **kwargs) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        host=HOST, client=client if client is not None else FakeHTTP(), **kwargs
    )


def test_space_is_the_registered_nomic_space():
    assert build().space is NOMIC_SPACE


def test_space_is_768_wide():
    assert build().space.dimensions == 768


def test_nomic_has_its_own_collection():
    assert build().space.collection == "knowledge--nomic-768"


def test_nomic_does_not_share_geminis_collection():
    assert build().space.collection != "knowledge"


def test_endpoint_is_the_ollama_embed_route():
    http = FakeHTTP()
    build(http).embed_documents(["hello"])

    assert http.calls[0]["url"] == f"{HOST}/api/embed"


def test_request_carries_the_model():
    http = FakeHTTP()
    build(http).embed_documents(["hello"])

    assert http.calls[0]["json"]["model"] == "nomic-embed-text-v2-moe"


def test_request_does_not_send_dimensions():
    """nomic is natively 768-dim; asking for a width it cannot produce fails."""
    http = FakeHTTP()
    build(http).embed_documents(["hello"])

    assert "dimensions" not in http.calls[0]["json"]


def test_request_carries_keep_alive():
    http = FakeHTTP()
    build(http, keep_alive="10m").embed_documents(["hello"])

    assert http.calls[0]["json"]["keep_alive"] == "10m"


def test_default_keep_alive_is_thirty_minutes():
    http = FakeHTTP()
    build(http).embed_documents(["hello"])

    assert http.calls[0]["json"]["keep_alive"] == "30m"


def test_documents_carry_the_document_prefix():
    http = FakeHTTP()
    build(http).embed_documents(["what is a transformer"])

    assert http.calls[0]["json"]["input"] == [
        f"{DOCUMENT_INSTRUCTION}what is a transformer"
    ]


def test_queries_carry_the_query_prefix():
    http = FakeHTTP()
    build(http).embed_query("what is a transformer")

    assert http.calls[0]["json"]["input"] == [
        f"{QUERY_INSTRUCTION}what is a transformer"
    ]


def test_the_two_prefixes_differ():
    assert QUERY_INSTRUCTION != DOCUMENT_INSTRUCTION


def test_every_document_in_a_batch_is_prefixed():
    http = FakeHTTP()
    build(http, batch_size=4).embed_documents(["a", "b", "c"])

    assert http.calls[0]["json"]["input"] == [
        f"{DOCUMENT_INSTRUCTION}a",
        f"{DOCUMENT_INSTRUCTION}b",
        f"{DOCUMENT_INSTRUCTION}c",
    ]


def test_one_request_per_batch():
    http = FakeHTTP()
    build(http, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])

    assert [len(call["json"]["input"]) for call in http.calls] == [2, 2, 1]


def test_batching_preserves_input_order():
    http = FakeHTTP(
        responses=[
            FakeResponse({"embeddings": [vec(1.0), vec(2.0)]}),
            FakeResponse({"embeddings": [vec(3.0), vec(4.0)]}),
            FakeResponse({"embeddings": [vec(5.0)]}),
        ]
    )

    vectors = build(http, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])

    assert vectors == [vec(1.0), vec(2.0), vec(3.0), vec(4.0), vec(5.0)]


def test_default_batch_size_is_sixty_four():
    assert DEFAULT_OLLAMA_BATCH_SIZE == 64


def test_empty_text_yields_no_request():
    http = FakeHTTP()

    assert build(http).embed_documents([]) == []
    assert http.calls == []


def test_blank_query_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="Query text"):
        build().embed_query("   ")


def test_short_embedding_count_is_reported():
    http = FakeHTTP(responses=[FakeResponse({"embeddings": [vec(1.0)]})])

    with pytest.raises(EmbeddingError, match="Expected 2 embeddings"):
        build(http).embed_documents(["a", "b"])


def test_ragged_vectors_are_reported():
    http = FakeHTTP(responses=[FakeResponse({"embeddings": [vec(1.0), vec(2.0)[:10]]})])

    with pytest.raises(EmbeddingError, match="Inconsistent embedding size"):
        build(http).embed_documents(["a", "b"])


def test_empty_embedding_is_reported():
    http = FakeHTTP(responses=[FakeResponse({"embeddings": [[]]})])

    with pytest.raises(EmbeddingError, match="empty embedding"):
        build(http).embed_documents(["a"])


def test_missing_embeddings_field_is_reported():
    http = FakeHTTP(responses=[FakeResponse({})])

    with pytest.raises(EmbeddingError, match="Expected 1 embeddings"):
        build(http).embed_documents(["a"])


def test_an_unexpected_width_is_refused():
    """A silent width change would corrupt the collection it is bound to."""
    http = FakeHTTP(responses=[FakeResponse({"embeddings": [[0.1] * 3072]})])

    with pytest.raises(EmbeddingError, match="registered as 768-dim"):
        build(http).embed_documents(["a"])


def test_http_error_becomes_an_embedding_api_error():
    http = FakeHTTP(responses=[FakeResponse(status=500)])

    with pytest.raises(EmbeddingAPIError, match="failed"):
        build(http).embed_documents(["a"])


def test_transport_failure_preserves_its_cause():
    class Broken:
        def post(self, *args, **kwargs):
            raise ConnectionRefusedError("connection refused")

    with pytest.raises(EmbeddingAPIError) as excinfo:
        build(Broken()).embed_documents(["a"])

    assert isinstance(excinfo.value.__cause__, ConnectionRefusedError)


def test_error_names_the_host_and_model():
    class Broken:
        def post(self, *args, **kwargs):
            raise ConnectionRefusedError("nope")

    with pytest.raises(EmbeddingAPIError) as excinfo:
        build(Broken()).embed_documents(["a"])

    message = str(excinfo.value)
    assert HOST in message
    assert "nomic-embed-text-v2-moe" in message


def test_host_is_required(monkeypatch):
    monkeypatch.setattr("knowledge_agent.embeddings.ollama.OLLAMA_HOST", None)

    with pytest.raises(EmbeddingConfigError, match="No Ollama host"):
        OllamaEmbeddingProvider(client=FakeHTTP())


def test_trailing_slash_is_normalised():
    http = FakeHTTP()
    OllamaEmbeddingProvider(host=f"{HOST}/", client=http).embed_documents(["a"])

    assert http.calls[0]["url"] == f"{HOST}/api/embed"


def test_batch_size_must_be_positive():
    with pytest.raises(EmbeddingConfigError, match="batch_size"):
        build(batch_size=0)


def test_timeout_must_be_positive():
    with pytest.raises(EmbeddingConfigError, match="timeout"):
        build(timeout=0)


def test_timeout_can_be_overridden_by_the_environment(monkeypatch):
    monkeypatch.setattr("knowledge_agent.embeddings.ollama.OLLAMA_TIMEOUT", "900")

    assert build().timeout == 900.0


def test_an_explicit_timeout_wins_over_the_environment(monkeypatch):
    monkeypatch.setattr("knowledge_agent.embeddings.ollama.OLLAMA_TIMEOUT", "900")

    assert build(timeout=30).timeout == 30.0


def test_a_non_numeric_timeout_is_rejected(monkeypatch):
    monkeypatch.setattr("knowledge_agent.embeddings.ollama.OLLAMA_TIMEOUT", "soon")

    with pytest.raises(EmbeddingConfigError, match="must be a number"):
        build()


def test_explicit_space_is_kept_verbatim():
    provider = OllamaEmbeddingProvider(
        host=HOST, client=FakeHTTP(), space=NOMIC_SPACE, model="custom-embed"
    )

    assert provider.space is NOMIC_SPACE
