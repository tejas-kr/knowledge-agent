import importlib
import sys
import tomllib
from pathlib import Path

import pytest

from knowledge_agent import main as main_module
from knowledge_agent.embeddings.resolver import Resolution
from knowledge_agent.embeddings.space import GEMINI_SPACE, NOMIC_SPACE, SPACES
from knowledge_agent.ingestion.pipeline import IngestReport
from knowledge_agent.ingestion.loader import DocumentNotFoundError
from knowledge_agent.main import build_parser, main
from knowledge_agent.rag.generator import GenerationConfigError
from knowledge_agent.rag.pipeline import RAGError, build_citations
from knowledge_agent.retrieval.vector_store import VectorStoreError
from knowledge_agent.schemas.rag import RAGAnswer
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

    def existing_ids(self):
        return set()


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(
        main_module,
        "build_store",
        lambda args, space=None: main_module.StoreHandle(store=fake, space=GEMINI_SPACE),
    )
    monkeypatch.setattr(main_module, "ChromaVectorStore", lambda **kwargs: fake)
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


def test_parser_resume_defaults_to_false():
    assert build_parser().parse_args(["index"]).resume is False


def test_parser_reads_resume_flag():
    assert build_parser().parse_args(["index", "--resume"]).resume is True


def test_index_forwards_resume(monkeypatch, store):
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return IngestReport(documents=1, pages=1, total_chunks=1)

    monkeypatch.setattr(main_module, "index_directory", capture)

    main(["index", "--resume"])

    assert seen["skip_existing"] is True


def test_index_without_resume_does_not_skip(monkeypatch, store):
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return IngestReport(documents=1, pages=1, total_chunks=1)

    monkeypatch.setattr(main_module, "index_directory", capture)

    main(["index"])

    assert seen["skip_existing"] is False


def test_index_reports_skipped_chunks(monkeypatch, store, capsys):
    patch_index(
        monkeypatch,
        IngestReport(documents=1, pages=2, chunks_indexed=5, chunks_skipped=819, total_chunks=824),
    )

    main(["index", "--resume"])

    assert "Skipped 819" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    "argv", [["index"], ["search", "q"], ["count"]]
)
def test_space_flag_defaults_to_none(argv):
    assert build_parser().parse_args(argv).space is None


@pytest.mark.parametrize("command", ["index", "count"])
def test_space_flag_is_read(command):
    assert build_parser().parse_args([command, "--space", "nomic-768"]).space == "nomic-768"


def test_search_space_flag_is_read():
    args = build_parser().parse_args(["search", "q", "--space", "gemini-3072"])

    assert args.space == "gemini-3072"


def test_an_unknown_space_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["search", "q", "--space", "nope"])


def test_spaces_is_a_command():
    assert build_parser().parse_args(["spaces"]).command == "spaces"


def test_compare_reads_query_and_k():
    args = build_parser().parse_args(["compare", "why", "-k", "4"])

    assert (args.query, args.k) == ("why", 4)


def test_index_announces_the_space_and_collection(monkeypatch, store, capsys):
    patch_index(monkeypatch, IngestReport())

    main(["index"])

    out = capsys.readouterr().out
    assert "space=gemini-3072" in out
    assert "collection=knowledge" in out


def test_count_names_the_space_and_collection(store, capsys):
    store.total = 7

    main(["count"])

    out = capsys.readouterr().out
    assert "gemini-3072" in out
    assert "7 chunks" in out
    assert "knowledge" in out


def test_search_survives_characters_the_console_cannot_encode(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module,
        "build_store",
        lambda args, space=None: main_module.StoreHandle(
            store=FakeStore(results=[make_result(content="keys, values ⨆ queries")]),
            space=GEMINI_SPACE,
        ),
    )

    assert main(["search", "q"]) == 0
    assert "⨆" in capsys.readouterr().out


def test_output_streams_are_switched_to_utf8():
    class Stream:
        def __init__(self):
            self.calls = []

        def reconfigure(self, **kwargs):
            self.calls.append(kwargs)

    out, err = Stream(), Stream()
    patch = pytest.MonkeyPatch()
    patch.setattr(main_module.sys, "stdout", out)
    patch.setattr(main_module.sys, "stderr", err)

    try:
        main_module._use_utf8_output()
    finally:
        patch.undo()

    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_a_stream_without_reconfigure_is_skipped():
    class Bare:
        pass

    patch = pytest.MonkeyPatch()
    patch.setattr(main_module.sys, "stdout", Bare())
    patch.setattr(main_module.sys, "stderr", Bare())

    try:
        main_module._use_utf8_output()
    finally:
        patch.undo()


def test_spaces_lists_every_space(store, capsys):
    store.total = 11

    assert main(["spaces"]) == 0

    out = capsys.readouterr().out
    assert "gemini-3072" in out
    assert "nomic-768" in out
    assert "gemini-embedding-001" in out
    assert "nomic-embed-text-v2-moe" in out
    assert "knowledge--nomic-768" in out
    assert "11" in out


def test_build_store_trusts_the_resolver_over_the_provider(monkeypatch):
    class MislabelledProvider:
        space = NOMIC_SPACE

    resolution = Resolution(provider=MislabelledProvider(), space=GEMINI_SPACE)
    monkeypatch.setattr(main_module, "resolve_provider", lambda preferred=None: resolution)

    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return FakeStore()

    monkeypatch.setattr(main_module, "ChromaVectorStore", record)

    handle = main_module.build_store(build_parser().parse_args(["search", "q"]))

    assert seen["space"] is GEMINI_SPACE
    assert handle.space is GEMINI_SPACE


def test_spaces_needs_no_provider(monkeypatch, capsys):
    seen = []

    def record(**kwargs):
        seen.append(kwargs)
        return FakeStore(total=0)

    monkeypatch.setattr(main_module, "ChromaVectorStore", record)

    assert main(["spaces"]) == 0
    assert seen and all("provider" not in kwargs for kwargs in seen)
    assert "nomic-768" in capsys.readouterr().out


def test_search_reports_a_store_mismatch(monkeypatch, capsys):
    def refuse(args, space=None):
        raise VectorStoreError("cannot share an index")

    monkeypatch.setattr(main_module, "build_store", refuse)

    assert main(["search", "q"]) == 1
    assert "cannot share an index" in capsys.readouterr().err


@pytest.fixture
def compare_store(monkeypatch):
    gemini = FakeStore(results=[make_result(chunk_id="g:1:0", distance=0.30)])
    nomic = FakeStore(results=[make_result(chunk_id="q:1:0", distance=0.10)])
    handles = {"gemini-3072": gemini, "nomic-768": nomic}

    monkeypatch.setattr(
        main_module,
        "build_store",
        lambda args, space=None: main_module.StoreHandle(
            store=handles[space or "gemini-3072"], space=SPACES[space or "gemini-3072"]
        ),
    )

    return handles


def test_compare_labels_every_result_with_its_space(compare_store, capsys):
    assert main(["compare", "question"]) == 0

    out = capsys.readouterr().out
    assert "space=gemini-3072" in out
    assert "space=nomic-768" in out


def test_compare_orders_by_distance(compare_store, capsys):
    main(["compare", "question"])

    lines = [line for line in capsys.readouterr().out.splitlines() if "distance=" in line]

    assert "nomic-768" in lines[0]
    assert "gemini-3072" in lines[1]


def test_compare_warns_against_comparing_distances(compare_store, capsys):
    main(["compare", "question"])

    assert "not comparable" in capsys.readouterr().out


def test_compare_embeds_the_query_once_per_space(compare_store):
    main(["compare", "question"])

    assert [store.searches for store in compare_store.values()] == [
        [("question", 5, None)],
        [("question", 5, None)],
    ]


def test_compare_skips_an_empty_collection(monkeypatch, capsys):
    empty = FakeStore(results=[])
    full = FakeStore(results=[make_result(chunk_id="q:1:0")])
    handles = {"gemini-3072": empty, "nomic-768": full}
    monkeypatch.setattr(
        main_module,
        "build_store",
        lambda args, space=None: main_module.StoreHandle(
            store=handles[space], space=SPACES[space]
        ),
    )

    assert main(["compare", "question"]) == 0

    assert "is empty" in capsys.readouterr().err


def test_compare_skips_an_unconstructible_space(monkeypatch, capsys):
    def build(args, space=None):
        if space == "gemini-3072":
            raise VectorStoreError("held a different embedding space")
        return main_module.StoreHandle(
            store=FakeStore(results=[make_result(chunk_id="q:1:0")]), space=SPACES["nomic-768"]
        )

    monkeypatch.setattr(main_module, "build_store", build)

    assert main(["compare", "question"]) == 0
    assert "skipped gemini-3072" in capsys.readouterr().err


def test_compare_returns_one_when_nothing_matched(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "build_store",
        lambda args, space=None: main_module.StoreHandle(
            store=FakeStore(results=[]), space=SPACES[space or "gemini-3072"]
        ),
    )

    assert main(["compare", "question"]) == 1


def test_ask_parser_reads_the_question():
    args = build_parser().parse_args(["ask", "what is attention?"])

    assert args.question == "what is attention?"


def test_ask_defaults_to_the_complete_index():
    args = build_parser().parse_args(["ask", "question"])

    assert args.space == "nomic-768"
    assert args.k == 5
    assert args.min_distance is None


def test_ask_parser_reads_k_and_min_distance():
    args = build_parser().parse_args(["ask", "q", "-k", "3", "--min-distance", "0.4"])

    assert args.k == 3
    assert args.min_distance == 0.4


def test_ask_parser_accepts_a_space_pin():
    assert build_parser().parse_args(["ask", "q", "--space", "gemini-3072"]).space == "gemini-3072"


def test_serve_parser_defaults_to_localhost_8000():
    args = build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_serve_parser_reads_host_and_port():
    args = build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "9001"])

    assert args.host == "0.0.0.0"
    assert args.port == 9001


def fake_generator(text="grounded answer [1]"):
    return type(
        "FakeGenerator", (), {"__init__": lambda self: None, "generate": lambda self, p: text}
    )()


def patch_ask(monkeypatch, text="grounded answer [1]", results=None, error=None):
    seen = {}

    def run(question, store, generator, space=None, k=5, min_distance=None, **kwargs):
        seen.update(question=question, store=store, k=k, min_distance=min_distance, space=space)

        if error:
            raise error

        answer = RAGAnswer(
            question=question,
            answer=text,
            citations=build_citations(results or [], text),
            space=space.key if space else "",
        )
        return answer

    monkeypatch.setattr(main_module, "build_generator", fake_generator)
    monkeypatch.setattr(main_module, "answer_question", run)
    return seen


def test_ask_prints_the_answer(monkeypatch, store, capsys):
    patch_ask(monkeypatch)

    assert main(["ask", "what is attention?"]) == 0

    assert "grounded answer [1]" in capsys.readouterr().out


def test_ask_prints_sources_with_filename_and_page(monkeypatch, store, capsys):
    patch_ask(monkeypatch, results=[make_result(filename="1706.03762v7.pdf", page_number=5)])

    main(["ask", "question"])

    out = capsys.readouterr().out

    assert "1706.03762v7.pdf" in out
    assert "p5" in out


def test_ask_marks_uncited_sources_as_unused(monkeypatch, store, capsys):
    patch_ask(monkeypatch, text="no markers here", results=[make_result()])

    main(["ask", "question"])

    assert "unused" in capsys.readouterr().out


def test_ask_says_so_when_nothing_is_cited(monkeypatch, store, capsys):
    patch_ask(monkeypatch, text="no context at all", results=[])

    assert main(["ask", "question"]) == 0

    assert "Sources: none" in capsys.readouterr().err


def test_ask_forwards_the_question_and_k(monkeypatch, store):
    seen = patch_ask(monkeypatch)

    main(["ask", "why?", "-k", "3"])

    assert seen["question"] == "why?"
    assert seen["k"] == 3


def test_ask_forwards_min_distance(monkeypatch, store):
    seen = patch_ask(monkeypatch)

    main(["ask", "q", "--min-distance", "0.4"])

    assert seen["min_distance"] == 0.4


def test_ask_reports_the_space_it_used(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module,
        "build_store",
        lambda args, space=None: main_module.StoreHandle(
            store=FakeStore(results=[make_result()]), space=NOMIC_SPACE
        ),
    )
    patch_ask(monkeypatch)

    main(["ask", "question"])

    assert "nomic-768" in capsys.readouterr().out


def test_ask_json_output_is_machine_readable(monkeypatch, store, capsys):
    import json

    patch_ask(monkeypatch, results=[make_result()])

    assert main(["ask", "question", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["answer"] == "grounded answer [1]"
    assert payload["citations"][0]["filename"] == "doc.pdf"


def test_ask_returns_one_on_a_pipeline_failure(monkeypatch, store, capsys):
    patch_ask(monkeypatch, error=RAGError("Retrieval failed: chroma down"))

    assert main(["ask", "question"]) == 1
    assert "Ask failed" in capsys.readouterr().err


def test_ask_returns_one_when_generation_is_unconfigured(monkeypatch, store, capsys):
    def refuse():
        raise GenerationConfigError("No API key")

    monkeypatch.setattr(main_module, "build_generator", refuse)
    monkeypatch.setattr(main_module, "answer_question", lambda *a, **kw: None)

    assert main(["ask", "question"]) == 1
    assert "No API key" in capsys.readouterr().err


def test_serve_starts_uvicorn_on_the_requested_port(monkeypatch, capsys):
    started = {}

    class FakeUvicorn:
        @staticmethod
        def run(target, host, port):
            started.update(target=target, host=host, port=port)

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)

    assert main(["serve", "--port", "9001"]) == 0

    assert started == {
        "target": "knowledge_agent.api:app",
        "host": "127.0.0.1",
        "port": 9001,
    }


def test_serve_announces_the_endpoint(monkeypatch, capsys):
    class FakeUvicorn:
        @staticmethod
        def run(target, host, port):
            return None

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)

    main(["serve"])

    assert "/ask" in capsys.readouterr().out


def test_serve_reports_a_missing_uvicorn(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def no_uvicorn(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_uvicorn)

    assert main(["serve"]) == 1
    assert "uvicorn" in capsys.readouterr().err