import importlib
import tomllib
from pathlib import Path

import pytest

from knowledge_agent import main as main_module
from knowledge_agent.ingestion.pipeline import IngestReport
from knowledge_agent.ingestion.loader import DocumentNotFoundError
from knowledge_agent.main import build_parser, main
from knowledge_agent.retrieval.vector_store import VectorStoreError
from knowledge_agent.schemas.retrieval import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    def __init__(self, total: int = 0, results=None, error: Exception | None = None):
        self.total = total
        self.results = results or []
        self.error = error
        self.searches: list[tuple] = []
        self.collection_name = "knowledge"

    def count(self):
        return self.total

    def search(self, query, k=5, where=None):
        self.searches.append((query, k, where))
        return self.results

    def add_chunks(self, chunks, embeddings=None):
        return []


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(main_module, "ChromaVectorStore", lambda **kwargs: fake)
    monkeypatch.setattr(main_module, "GeminiEmbeddingProvider", lambda **kwargs: object())
    return fake


def make_result(**kwargs) -> SearchResult:
    defaults = {
        "chunk_id": "doc:1:0",
        "content": "some content",
        "distance": 0.25,
        "document_id": "doc",
        "filename": "doc.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }
    defaults.update(kwargs)
    return SearchResult(**defaults)


def patch_index(monkeypatch, report: IngestReport):
    monkeypatch.setattr(main_module, "index_directory", lambda *a, **kw: report)


def test_parser_defaults_to_index_when_no_command():
    assert build_parser().parse_args([]).command is None


def test_parser_reads_index_command():
    assert build_parser().parse_args(["index"]).command == "index"


def test_parser_reads_search_query_and_k():
    args = build_parser().parse_args(["search", "what is rag", "-k", "3"])

    assert args.query == "what is rag"
    assert args.k == 3


def test_parser_reads_document_filter():
    assert build_parser().parse_args(["search", "q", "--document", "d"]).document == "d"


def test_no_arguments_runs_index(monkeypatch, store, capsys):
    patch_index(monkeypatch, IngestReport(documents=1, pages=2, chunks_indexed=3, total_chunks=3))

    assert main([]) == 0
    assert "Indexed 3 chunks" in capsys.readouterr().out


def test_index_prints_totals(monkeypatch, store, capsys):
    patch_index(monkeypatch, IngestReport(documents=2, pages=9, chunks_indexed=20, total_chunks=55))

    main(["index"])

    out = capsys.readouterr().out
    assert "20 chunks" in out
    assert "55 chunks" in out


def test_index_returns_one_on_load_failure(monkeypatch, store, capsys):
    patch_index(
        monkeypatch,
        IngestReport(load_failures=[DocumentNotFoundError("broken.pdf")]),
    )

    assert main(["index"]) == 1
    assert "Failed to load" in capsys.readouterr().err


def test_index_returns_one_on_store_failure(monkeypatch, store, capsys):
    patch_index(monkeypatch, IngestReport(store_failures=[VectorStoreError("chroma down")]))

    assert main(["index"]) == 1
    assert "Failed to store" in capsys.readouterr().err


def test_count_prints_chunk_total(store, capsys):
    store.total = 42

    assert main(["count"]) == 0
    assert "42 chunks" in capsys.readouterr().out


def test_search_prints_ranked_results(store, capsys):
    store.results = [make_result(chunk_id="doc:1:0"), make_result(chunk_id="doc:1:1")]

    assert main(["search", "question"]) == 0

    out = capsys.readouterr().out
    assert "1. [doc.pdf p1]" in out
    assert "2. [doc.pdf p1]" in out
    assert "distance=0.2500" in out
    assert "some content" in out


def test_search_passes_query_and_k(store):
    main(["search", "question", "-k", "2"])

    assert store.searches == [("question", 2, None)]


def test_search_passes_document_filter(store):
    main(["search", "question", "--document", "doc"])

    assert store.searches == [("question", 5, {"document_id": "doc"})]


def test_search_with_no_matches_says_so(store, capsys):
    store.results = []

    assert main(["search", "question"]) == 0
    assert "No matching chunks" in capsys.readouterr().out


def test_similarity_is_one_minus_distance():
    assert make_result(distance=0.25).similarity == pytest.approx(0.75)


def test_console_script_entry_point_resolves_to_callable():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["knowledge-agent"]

    module_name, _, attribute = target.partition(":")

    assert attribute, f"entry point {target!r} must use the 'module:attribute' form"
    assert callable(getattr(importlib.import_module(module_name), attribute))
