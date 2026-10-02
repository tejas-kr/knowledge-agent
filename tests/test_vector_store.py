import pytest

from knowledge_agent.embeddings.provider import EmbeddingProvider, EmbeddingQuotaError
from knowledge_agent.embeddings.space import GEMINI_SPACE, NOMIC_SPACE, EmbeddingSpace
from knowledge_agent.retrieval.vector_store import (
    DEFAULT_COLLECTION_NAME,
    EMBEDDING_SPACE_KEY,
    ChromaVectorStore,
    VectorStore,
    VectorStoreError,
)
from knowledge_agent.schemas.chunk import Chunk

VOCABULARY = "abcdefghijklmnopqrstuvwxyz"


class StubProvider(EmbeddingProvider):
    """Deterministic bag-of-characters embeddings, so cosine order is meaningful."""

    def __init__(self, space: EmbeddingSpace | None = None) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self._space = space if space is not None else GEMINI_SPACE

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def embed_documents(self, texts):
        items = list(texts)
        self.document_calls.append(items)
        return [self._vector(t) for t in items]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(letter)) for letter in VOCABULARY]


def make_chunk(content: str, document_id: str = "doc", page_number: int = 1, chunk_index: int = 0) -> Chunk:
    return Chunk(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        content=content,
    )


class FakeCollection:
    def __init__(
        self,
        error: Exception | None = None,
        response: dict | None = None,
        count=0,
        metadata: dict | None = None,
    ):
        self.error = error
        self.response = response or {
            "ids": [["doc:1:0"]],
            "documents": [["stored text"]],
            "metadatas": [[{
                "document_id": "doc",
                "filename": "doc.pdf",
                "page_number": 1,
                "chunk_index": 0,
            }]],
            "distances": [[0.25]],
        }
        self.total = count
        self.upsert_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.metadata = metadata

    def upsert(self, **kwargs):
        if self.error:
            raise self.error
        self.upsert_calls.append(kwargs)
        return None

    def query(self, **kwargs):
        if self.error:
            raise self.error
        self.query_calls.append(kwargs)
        return self.response

    def delete(self, **kwargs):
        if self.error:
            raise self.error
        self.delete_calls.append(kwargs)
        return None

    def count(self):
        if self.error:
            raise self.error
        return self.total


class FakeClient:
    def __init__(self, collection: FakeCollection | None = None):
        self.collection = collection or FakeCollection()
        self.create_calls: list[dict] = []

    def get_or_create_collection(self, **kwargs):
        self.create_calls.append(kwargs)

        if self.collection.metadata is None:
            self.collection.metadata = kwargs.get("metadata")

        return self.collection


def build(collection: FakeCollection | None = None, provider=None) -> ChromaVectorStore:
    client = FakeClient(collection)
    store = ChromaVectorStore(
        provider=provider or StubProvider(),
        directory="C:/tmp/chroma",
        client=client,
    )
    return store


def test_vector_store_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()


def test_chunk_id_format():
    assert make_chunk("x").chunk_id == "doc:1:0"


def test_chunk_id_distinguishes_pages_and_indexes():
    assert make_chunk("x", page_number=7, chunk_index=3).chunk_id == "doc:7:3"


def test_chunk_id_distinguishes_documents():
    assert make_chunk("x", document_id="other").chunk_id == "other:1:0"


def test_collection_created_with_cosine_space():
    store = build()

    create = store._client.create_calls[0]
    assert create["name"] == DEFAULT_COLLECTION_NAME
    assert create["metadata"]["hnsw:space"] == "cosine"


def test_collection_records_embedding_space_identity():
    store = build()

    create = store._client.create_calls[0]
    assert create["metadata"][EMBEDDING_SPACE_KEY] == "gemini-3072"


def test_each_space_owns_its_own_collection():
    client = FakeClient()

    ChromaVectorStore(StubProvider(NOMIC_SPACE), directory="C:/tmp", client=client)

    create = client.create_calls[0]
    assert create["name"] == NOMIC_SPACE.collection == "knowledge--nomic-768"
    assert create["metadata"][EMBEDDING_SPACE_KEY] == "nomic-768"


def test_collection_is_selected_from_the_space_without_a_name():
    store = ChromaVectorStore(space=NOMIC_SPACE, directory="C:/tmp", client=FakeClient())

    assert store.collection_name == "knowledge--nomic-768"


def test_a_store_needs_a_provider_or_a_space():
    with pytest.raises(VectorStoreError, match="provider or space"):
        ChromaVectorStore(directory="C:/tmp", client=FakeClient())


def test_mismatched_embedding_space_is_rejected(tmp_path):
    collection = FakeCollection(metadata={EMBEDDING_SPACE_KEY: "nomic-768"})

    with pytest.raises(VectorStoreError, match="cannot share an index"):
        ChromaVectorStore(StubProvider(GEMINI_SPACE), directory=tmp_path, client=FakeClient(collection))


def test_mismatch_names_the_space_to_retry(tmp_path):
    collection = FakeCollection(metadata={EMBEDDING_SPACE_KEY: "nomic-768"})

    with pytest.raises(VectorStoreError, match="--space nomic-768"):
        ChromaVectorStore(StubProvider(GEMINI_SPACE), directory=tmp_path, client=FakeClient(collection))


def test_legacy_collection_without_space_metadata_is_accepted(tmp_path):
    collection = FakeCollection(metadata={"hnsw:space": "cosine"})

    store = ChromaVectorStore(StubProvider(GEMINI_SPACE), directory=tmp_path, client=FakeClient(collection))

    assert store.collection_name == DEFAULT_COLLECTION_NAME


def test_legacy_collection_is_owned_by_gemini_regardless_of_width(tmp_path):
    collection = FakeCollection(metadata={"hnsw:space": "cosine"})

    with pytest.raises(VectorStoreError, match="cannot share an index"):
        ChromaVectorStore(
            StubProvider(NOMIC_SPACE), directory=tmp_path, collection_name="knowledge",
            client=FakeClient(collection),
        )


def test_legacy_collection_points_at_its_owner(tmp_path):
    collection = FakeCollection(metadata={"hnsw:space": "cosine"})

    with pytest.raises(VectorStoreError, match="--space gemini-3072"):
        ChromaVectorStore(
            StubProvider(NOMIC_SPACE), directory=tmp_path, collection_name="knowledge",
            client=FakeClient(collection),
        )


def test_an_unrecognised_collection_has_no_owner_to_contradict(tmp_path):
    collection = FakeCollection(metadata={"hnsw:space": "cosine"})

    store = ChromaVectorStore(
        StubProvider(NOMIC_SPACE), directory=tmp_path, collection_name="scratch",
        client=FakeClient(collection),
    )

    assert store.collection_name == "scratch"


def test_counting_does_not_need_a_provider(tmp_path):
    store = ChromaVectorStore(space=NOMIC_SPACE, directory=tmp_path, client=FakeClient())

    assert store.count() == 0


def test_search_without_a_provider_reports_the_gap(tmp_path):
    store = ChromaVectorStore(
        space=NOMIC_SPACE,
        directory=tmp_path,
        client=FakeClient(FakeCollection(count=3)),
    )

    with pytest.raises(VectorStoreError, match="No embedding provider"):
        store.search("question")


def test_query_embedding_failure_is_wrapped(tmp_path):
    class FailingProvider(StubProvider):
        def embed_query(self, text):
            raise EmbeddingQuotaError("quota exhausted")

    store = ChromaVectorStore(
        FailingProvider(), directory=tmp_path, client=FakeClient(FakeCollection(count=3))
    )

    with pytest.raises(VectorStoreError, match="Failed to embed query"):
        store.search("question")


def test_custom_collection_name_is_used():
    store = ChromaVectorStore(
        provider=StubProvider(), directory="C:/tmp/chroma", collection_name="custom",
        client=FakeClient(),
    )

    assert store._client.create_calls[0]["name"] == "custom"


def test_empty_collection_name_is_rejected():
    with pytest.raises(VectorStoreError, match="collection_name"):
        ChromaVectorStore(
            provider=StubProvider(), directory="C:/tmp", collection_name="  ", client=FakeClient()
        )


def test_directory_defaults_to_config_chroma_dir():
    from knowledge_agent.config import CHROMA_DIR

    store = ChromaVectorStore(provider=StubProvider(), client=FakeClient())

    assert store.directory == CHROMA_DIR


def test_add_chunks_returns_ids():
    store = build()

    ids = store.add_chunks([make_chunk("a"), make_chunk("b", chunk_index=1)])

    assert ids == ["doc:1:0", "doc:1:1"]


def test_add_chunks_embeds_contents():
    provider = StubProvider()
    store = build(provider=provider)

    store.add_chunks([make_chunk("alpha"), make_chunk("beta", chunk_index=1)])

    assert provider.document_calls == [["alpha", "beta"]]


def test_add_chunks_passes_documents_and_metadata():
    store = build()

    store.add_chunks([make_chunk("alpha")])

    call = store._collection.upsert_calls[0]
    assert call["ids"] == ["doc:1:0"]
    assert call["documents"] == ["alpha"]
    assert call["metadatas"] == [{
        "document_id": "doc",
        "filename": "doc.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }]


def test_add_chunks_passes_embeddings():
    store = build()

    store.add_chunks([make_chunk("alpha")], embeddings=[[0.1, 0.2]])

    assert store._collection.upsert_calls[0]["embeddings"] == [[0.1, 0.2]]


def test_add_chunks_with_embeddings_skips_the_provider():
    provider = StubProvider()
    store = build(provider=provider)

    store.add_chunks([make_chunk("alpha")], embeddings=[[0.1]])

    assert provider.document_calls == []


def test_embedding_count_mismatch_is_rejected():
    store = build()

    with pytest.raises(VectorStoreError, match="Expected 2 embeddings, got 1"):
        store.add_chunks([make_chunk("a"), make_chunk("b", chunk_index=1)], embeddings=[[0.1]])


def test_add_chunks_on_empty_input_does_not_call_chroma():
    store = build()

    assert store.add_chunks([]) == []
    assert store._collection.upsert_calls == []


def test_upsert_failure_is_wrapped():
    store = build(FakeCollection(error=RuntimeError("disk full")))

    with pytest.raises(VectorStoreError, match="disk full"):
        store.add_chunks([make_chunk("a")])


def test_upsert_failure_preserves_cause():
    cause = RuntimeError("boom")
    store = build(FakeCollection(error=cause))

    with pytest.raises(VectorStoreError) as info:
        store.add_chunks([make_chunk("a")])

    assert info.value.__cause__ is cause


def test_search_embeds_the_query():
    provider = StubProvider()
    store = build(provider=provider, collection=FakeCollection(count=5))

    store.search("question")

    assert provider.query_calls == ["question"]


def test_search_passes_vector_and_k():
    store = build(collection=FakeCollection(count=5))

    store.search("question", k=3)

    call = store._collection.query_calls[0]
    assert call["n_results"] == 3
    assert call["query_embeddings"] == [provider_vector("question")]


def provider_vector(text: str) -> list[float]:
    return StubProvider._vector(text)


def test_search_maps_results():
    store = build(collection=FakeCollection(count=5))

    results = store.search("question")

    assert len(results) == 1
    assert results[0].chunk_id == "doc:1:0"
    assert results[0].content == "stored text"
    assert results[0].distance == 0.25
    assert results[0].document_id == "doc"
    assert results[0].filename == "doc.pdf"
    assert results[0].page_number == 1
    assert results[0].chunk_index == 0


def test_similarity_is_one_minus_distance():
    store = build(collection=FakeCollection(count=5))

    assert store.search("question")[0].similarity == pytest.approx(0.75)


def test_search_passes_where_filter():
    store = build(collection=FakeCollection(count=5))

    store.search("question", where={"document_id": "doc"})

    assert store._collection.query_calls[0]["where"] == {"document_id": "doc"}


def test_search_without_filter_sends_none():
    store = build(collection=FakeCollection(count=5))

    store.search("question")

    assert store._collection.query_calls[0]["where"] is None


def test_search_on_empty_collection_returns_nothing():
    store = build(collection=FakeCollection(count=0))

    assert store.search("question") == []
    assert store._collection.query_calls == []


def test_search_rejects_blank_query():
    store = build(collection=FakeCollection(count=5))

    with pytest.raises(VectorStoreError, match="Query cannot be empty"):
        store.search("   ")


def test_search_rejects_non_positive_k():
    store = build(collection=FakeCollection(count=5))

    with pytest.raises(VectorStoreError, match="k must be positive"):
        store.search("question", k=0)


def test_search_failure_is_wrapped():
    store = build(FakeCollection(error=RuntimeError("bad filter")))

    with pytest.raises(VectorStoreError, match="bad filter"):
        store.search("question")


def test_search_tolerates_missing_response_keys():
    store = build(FakeCollection(count=5, response={"ids": [["doc:1:0"]]}))

    result = store.search("question")[0]

    assert result.chunk_id == "doc:1:0"
    assert result.content == ""
    assert result.distance == 0.0
    assert result.document_id == ""


def test_search_tolerates_null_metadatas():
    response = {
        "ids": [["doc:1:0"]],
        "documents": [["text"]],
        "metadatas": [[None]],
        "distances": [[0.1]],
    }
    store = build(FakeCollection(count=5, response=response))

    result = store.search("question")[0]

    assert result.page_number == 0
    assert result.document_id == ""


def test_search_handles_empty_response_lists():
    store = build(FakeCollection(count=5, response={"ids": [[]], "documents": [[]]}))

    assert store.search("question") == []


def test_delete_document_uses_where_clause():
    store = build()

    store.delete_document("doc")

    assert store._collection.delete_calls[0]["where"] == {"document_id": "doc"}


def test_delete_document_rejects_blank_id():
    store = build()

    with pytest.raises(VectorStoreError, match="document_id"):
        store.delete_document("  ")


def test_delete_failure_is_wrapped():
    store = build(FakeCollection(error=RuntimeError("locked")))

    with pytest.raises(VectorStoreError, match="locked"):
        store.delete_document("doc")


def test_count_delegates_to_chroma():
    store = build(FakeCollection(count=7))

    assert store.count() == 7


def test_count_failure_is_wrapped():
    store = build(FakeCollection(error=RuntimeError("timeout")))

    with pytest.raises(VectorStoreError, match="timeout"):
        store.count()


@pytest.fixture
def real_store(tmp_path):
    from knowledge_agent.config import CHROMA_DIR

    return ChromaVectorStore(
        provider=StubProvider(), directory=tmp_path / "chroma", collection_name="itest"
    )


def test_integration_add_and_count(real_store):
    assert real_store.count() == 0

    ids = real_store.add_chunks([make_chunk("alpha one", document_id="d1")])

    assert ids == ["d1:1:0"]
    assert real_store.count() == 1


def test_integration_search_returns_nearest_first(real_store):
    real_store.add_chunks([
        make_chunk("zebra stripes", document_id="d2"),
        make_chunk("dog bark", document_id="d2", chunk_index=1),
        make_chunk("cat purr", document_id="d2", chunk_index=2),
    ])

    results = real_store.search("dog", k=2)

    assert len(results) == 2
    assert results[0].document_id == "d2"
    assert results[0].content == "dog bark"
    assert results[0].distance <= results[1].distance


def test_integration_preserves_provenance(real_store):
    real_store.add_chunks([make_chunk("alpha", document_id="d3", page_number=4, chunk_index=2)])

    result = real_store.search("alpha", k=1)[0]

    assert result.chunk_id == "d3:4:2"
    assert result.document_id == "d3"
    assert result.filename == "d3.pdf"
    assert result.page_number == 4
    assert result.chunk_index == 2
    assert result.content == "alpha"


def test_integration_reupsert_replaces_same_chunk(real_store):
    real_store.add_chunks([make_chunk("first version", document_id="d4")])
    real_store.add_chunks([make_chunk("second version", document_id="d4")])

    assert real_store.count() == 1
    assert real_store.search("version", k=1)[0].content == "second version"


def test_integration_delete_only_removes_one_document(real_store):
    real_store.add_chunks([
        make_chunk("keep me", document_id="d5"),
        make_chunk("drop me", document_id="d6"),
    ])
    before = real_store.count()

    real_store.delete_document("d6")

    assert real_store.count() == before - 1
    assert {r.document_id for r in real_store.search("keep", k=5)} == {"d5"}


def test_integration_where_filter_by_document(real_store):
    real_store.add_chunks([
        make_chunk("alpha text", document_id="d7"),
        make_chunk("alpha text", document_id="d8"),
    ])

    results = real_store.search("alpha", k=5, where={"document_id": "d8"})

    assert results
    assert {r.document_id for r in results} == {"d8"}


def test_integration_where_filter_by_page_number(real_store):
    real_store.add_chunks([
        make_chunk("alpha text", document_id="d9", page_number=1),
        make_chunk("alpha text", document_id="d9", page_number=2),
    ])

    results = real_store.search("alpha", k=5, where={"page_number": 2})

    assert results
    assert {r.page_number for r in results} == {2}


def test_integration_k_limits_results(real_store):
    real_store.add_chunks([
        make_chunk("alpha one", document_id="d10", chunk_index=0),
        make_chunk("alpha two", document_id="d10", chunk_index=1),
        make_chunk("alpha three", document_id="d10", chunk_index=2),
    ])

    assert len(real_store.search("alpha", k=2)) == 2


def test_integration_search_on_empty_store_returns_nothing(tmp_path):
    store = ChromaVectorStore(
        provider=StubProvider(), directory=tmp_path / "empty", collection_name="itest"
    )

    assert store.search("anything") == []


def test_integration_persists_across_instances(tmp_path):
    directory = tmp_path / "persist"
    ChromaVectorStore(
        provider=StubProvider(), directory=directory, collection_name="keep"
    ).add_chunks([make_chunk("persisted text", document_id="dp")])

    reopened = ChromaVectorStore(
        provider=StubProvider(), directory=directory, collection_name="keep"
    )

    assert reopened.count() == 1
    assert reopened.search("persisted", k=1)[0].content == "persisted text"


def test_existing_ids_is_empty_on_a_new_store(tmp_path):
    store = ChromaVectorStore(
        provider=StubProvider(), directory=tmp_path / "fresh", collection_name="ids"
    )

    assert store.existing_ids() == set()


def test_existing_ids_reflects_stored_chunks(tmp_path):
    store = ChromaVectorStore(
        provider=StubProvider(), directory=tmp_path / "ids", collection_name="ids"
    )

    store.add_chunks([make_chunk("a", page_number=1), make_chunk("b", page_number=2)])

    assert store.existing_ids() == {"doc:1:0", "doc:2:0"}


def test_existing_ids_drops_after_delete(tmp_path):
    store = ChromaVectorStore(
        provider=StubProvider(), directory=tmp_path / "ids", collection_name="ids"
    )
    store.add_chunks([make_chunk("a", document_id="gone")])

    store.delete_document("gone")

    assert store.existing_ids() == set()


def test_embedding_failure_surfaces_as_store_error():
    from knowledge_agent.embeddings.provider import EmbeddingQuotaError

    class FailingProvider(StubProvider):
        def embed_documents(self, texts):
            raise EmbeddingQuotaError("daily cap reached")

    store = build(provider=FailingProvider())

    with pytest.raises(VectorStoreError, match="Failed to embed"):
        store.add_chunks([make_chunk("a")])


def test_embedding_failure_preserves_cause():
    from knowledge_agent.embeddings.provider import EmbeddingQuotaError

    cause = EmbeddingQuotaError("daily cap reached")

    class FailingProvider(StubProvider):
        def embed_documents(self, texts):
            raise cause

    store = build(provider=FailingProvider())

    with pytest.raises(VectorStoreError) as excinfo:
        store.add_chunks([make_chunk("a")])

    assert excinfo.value.__cause__ is cause


def test_empty_add_does_not_call_the_provider():
    provider = StubProvider()
    store = build(provider=provider)

    assert store.add_chunks([]) == []
    assert provider.document_calls == []


def test_existing_ids_reads_from_the_collection():
    class GetCollection(FakeCollection):
        def get(self, **kwargs):
            self.get_calls = getattr(self, "get_calls", [])
            self.get_calls.append(kwargs)
            return {"ids": ["a:1:0", "b:2:0"]}

    collection = GetCollection()
    store = build(collection=collection)

    assert store.existing_ids() == {"a:1:0", "b:2:0"}
    assert collection.get_calls[0]["include"] == []


def test_existing_ids_wraps_collection_failures():
    store = build(collection=FakeCollection(error=RuntimeError("chroma down")))

    with pytest.raises(VectorStoreError, match="Failed to read stored ids"):
        store.existing_ids()
