from pathlib import Path

import pymupdf
import pytest

from knowledge_agent.ingestion.chunker import RecursiveChunker
from knowledge_agent.ingestion.loader import DocumentNotFoundError
from knowledge_agent.ingestion.pipeline import (
    IngestReport,
    index_directory,
    load_documents,
)
from knowledge_agent.retrieval.vector_store import VectorStore, VectorStoreError

VOCABULARY = "abcdefghijklmnopqrstuvwxyz"


class StubProvider:
    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(letter)) for letter in VOCABULARY]


class FakeStore(VectorStore):
    """Models upsert semantics: re-adding the same id does not grow the store."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.records: dict[str, object] = {}
        self.add_calls = 0

    def add_chunks(self, chunks, embeddings=None):
        items = list(chunks)
        self.add_calls += 1

        if self.error is not None:
            raise self.error

        ids = [c.chunk_id for c in items]
        for chunk_id, chunk in zip(ids, items, strict=True):
            self.records[chunk_id] = chunk
        return ids

    def search(self, query, k=5, where=None):
        return []

    def delete_document(self, document_id):
        for key in [k for k, c in self.records.items() if c.document_id == document_id]:
            del self.records[key]

    def count(self):
        return len(self.records)

    def existing_ids(self):
        return set(self.records)


def make_pdf(path: Path, texts: list[str]) -> Path:
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_load_documents_reads_pdfs(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["page one", "page two"])

    pages, failures = load_documents(tmp_path)

    assert failures == []
    assert [p.page_number for p in pages] == [1, 2]
    assert [p.filename for p in pages] == ["a.pdf", "a.pdf"]


def test_load_documents_ignores_unsupported_files(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["text"])
    (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")
    (tmp_path / "sub.pdf").mkdir()

    pages, failures = load_documents(tmp_path)

    assert failures == []
    assert {p.filename for p in pages} == {"a.pdf"}


def test_load_documents_orders_documents_by_filename(tmp_path):
    for name in ("c.pdf", "a.pdf", "b.pdf"):
        make_pdf(tmp_path / name, [name])

    pages, _ = load_documents(tmp_path)

    assert [p.filename for p in pages] == ["a.pdf", "b.pdf", "c.pdf"]


def test_load_documents_isolates_a_corrupt_file(tmp_path):
    make_pdf(tmp_path / "good.pdf", ["intact"])
    (tmp_path / "broken.pdf").write_bytes(b"definitely not a pdf")

    pages, failures = load_documents(tmp_path)

    assert [p.filename for p in pages] == ["good.pdf"]
    assert len(failures) == 1
    assert "broken.pdf" in str(failures[0])


def test_load_documents_records_every_failure(tmp_path):
    for name in ("bad1.pdf", "bad2.pdf"):
        (tmp_path / name).write_bytes(b"not a pdf")

    pages, failures = load_documents(tmp_path)

    assert pages == []
    assert len(failures) == 2


def test_load_documents_on_empty_directory(tmp_path):
    assert load_documents(tmp_path) == ([], [])


def test_load_documents_missing_directory_raises(tmp_path):
    with pytest.raises(DocumentNotFoundError, match="nope"):
        load_documents(tmp_path / "nope")


def test_index_directory_stores_every_page(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha", "beta"])
    store = FakeStore()

    report = index_directory(tmp_path, store, chunker=RecursiveChunker(chunk_size=100))

    assert report.documents == 1
    assert report.pages == 2
    assert report.chunks_indexed == 2
    assert report.total_chunks == 2
    assert report.ok is True


def test_index_directory_splits_pages_into_chunks(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha beta gamma delta epsilon zeta"])
    store = FakeStore()

    report = index_directory(tmp_path, store, chunker=RecursiveChunker(chunk_size=15, chunk_overlap=0))

    assert report.chunks_indexed > 1
    assert store.add_calls == 1


def test_index_directory_uses_a_default_chunker(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha beta gamma"])
    store = FakeStore()

    report = index_directory(tmp_path, store)

    assert report.chunks_indexed >= 1
    assert report.ok is True


def test_index_directory_on_empty_directory(tmp_path):
    report = index_directory(tmp_path, FakeStore())

    assert report.documents == 0
    assert report.pages == 0
    assert report.chunks_indexed == 0
    assert report.ok is True


def test_index_directory_isolates_store_failures(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["one", "two"])
    store = FakeStore(error=VectorStoreError("chroma down"))

    report = index_directory(tmp_path, store)

    assert len(report.store_failures) == 1
    assert report.load_failures == []
    assert report.ok is False


def test_store_failure_is_recorded_once_per_batch(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["one", "two", "three", "four"])
    store = FakeStore(error=VectorStoreError("chroma down"))

    report = index_directory(tmp_path, store, batch_size=1)

    assert len(report.store_failures) == 4
    assert report.chunks_indexed == 0


def test_chunks_are_batched_across_pages(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha", "beta", "gamma", "delta"])
    store = FakeStore()

    index_directory(tmp_path, store, batch_size=2)

    assert store.add_calls == 2


def test_trailing_partial_batch_is_flushed(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha", "beta", "gamma"])
    store = FakeStore()

    report = index_directory(tmp_path, store, batch_size=2)

    assert store.add_calls == 2
    assert report.chunks_indexed == 3
    assert report.total_chunks == 3


def test_batching_reduces_api_round_trips(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["page"] * 20)
    store = FakeStore()

    report = index_directory(tmp_path, store)

    assert report.pages == 20
    assert report.chunks_indexed == 20
    assert store.add_calls == 1


def test_index_directory_reports_load_failures(tmp_path):
    make_pdf(tmp_path / "good.pdf", ["intact"])
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")

    report = index_directory(tmp_path, FakeStore())

    assert len(report.load_failures) == 1
    assert report.pages == 1
    assert report.ok is False


def test_reindexing_does_not_duplicate_chunks(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha", "beta"])
    store = FakeStore()

    first = index_directory(tmp_path, store, chunker=RecursiveChunker(chunk_size=100))
    second = index_directory(tmp_path, store, chunker=RecursiveChunker(chunk_size=100))

    assert first.total_chunks == 2
    assert second.total_chunks == 2
    assert second.chunks_indexed == 2


def test_resume_skips_already_stored_chunks(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha", "beta"])
    store = FakeStore()
    chunker = RecursiveChunker(chunk_size=100)

    index_directory(tmp_path, store, chunker=chunker)
    store.add_calls = 0

    resumed = index_directory(tmp_path, store, chunker=chunker, skip_existing=True)

    assert resumed.chunks_indexed == 0
    assert resumed.chunks_skipped == 2
    assert store.add_calls == 0
    assert resumed.total_chunks == 2


def test_resume_still_embeds_chunks_that_are_missing(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha"])
    store = FakeStore()
    chunker = RecursiveChunker(chunk_size=100)

    index_directory(tmp_path, store, chunker=chunker)

    make_pdf(tmp_path / "b.pdf", ["beta"])

    resumed = index_directory(tmp_path, store, chunker=chunker, skip_existing=True)

    assert resumed.chunks_skipped == 1
    assert resumed.chunks_indexed == 1
    assert resumed.total_chunks == 2


def test_resume_is_off_by_default(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["alpha"])
    store = FakeStore()
    chunker = RecursiveChunker(chunk_size=100)

    index_directory(tmp_path, store, chunker=chunker)

    again = index_directory(tmp_path, store, chunker=chunker)

    assert again.chunks_skipped == 0
    assert again.chunks_indexed == 1


def test_resume_against_real_chroma(tmp_path):
    from knowledge_agent.embeddings.provider import EmbeddingProvider
    from knowledge_agent.retrieval.vector_store import ChromaVectorStore

    class Provider(EmbeddingProvider):
        def __init__(self):
            self.calls = 0

        @property
        def space(self):
            from knowledge_agent.embeddings.space import GEMINI_SPACE

            return GEMINI_SPACE

        def embed_documents(self, texts):
            self.calls += 1
            return [StubProvider._vector(t) for t in texts]

        def embed_query(self, text):
            return StubProvider._vector(text)

    make_pdf(tmp_path / "a.pdf", ["alpha", "beta"])
    provider = Provider()
    store = ChromaVectorStore(
        provider=provider, directory=tmp_path / "chroma", collection_name="resume"
    )
    chunker = RecursiveChunker(chunk_size=100)

    index_directory(tmp_path, store, chunker=chunker)
    assert store.existing_ids() == {"a:1:0", "a:2:0"}

    resumed = index_directory(tmp_path, store, chunker=chunker, skip_existing=True)

    assert resumed.chunks_skipped == 2
    assert resumed.chunks_indexed == 0
    assert provider.calls == 1


def test_index_directory_missing_directory_raises(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        index_directory(tmp_path / "nope", FakeStore())


def test_report_ok_is_false_with_either_failure_type():
    assert IngestReport().ok is True
    assert IngestReport(store_failures=[VectorStoreError("x")]).ok is False


def test_index_directory_against_real_chroma(tmp_path):
    from knowledge_agent.embeddings.provider import EmbeddingProvider
    from knowledge_agent.retrieval.vector_store import ChromaVectorStore

    class Provider(EmbeddingProvider):
        @property
        def space(self):
            from knowledge_agent.embeddings.space import GEMINI_SPACE

            return GEMINI_SPACE

        def embed_documents(self, texts):
            return [StubProvider._vector(t) for t in texts]

        def embed_query(self, text):
            return StubProvider._vector(text)

    make_pdf(tmp_path / "a.pdf", ["alpha content", "beta content"])
    store = ChromaVectorStore(
        provider=Provider(), directory=tmp_path / "chroma", collection_name="pipe"
    )

    report = index_directory(tmp_path, store, chunker=RecursiveChunker(chunk_size=100))

    assert report.pages == 2
    assert report.chunks_indexed == 2
    assert report.total_chunks == 2
    assert store.search("alpha", k=1)[0].content == "alpha content"
