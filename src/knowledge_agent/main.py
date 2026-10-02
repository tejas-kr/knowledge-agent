import argparse
import json
import sys
from dataclasses import dataclass

from knowledge_agent.config import DOCUMENTS_DIR, RAG_TOP_K
from knowledge_agent.embeddings.provider import EmbeddingConfigError, EmbeddingError
from knowledge_agent.embeddings.resolver import resolve_collection, resolve_provider
from knowledge_agent.embeddings.space import SPACE_KEYS, SPACES, EmbeddingSpace
from knowledge_agent.ingestion.chunker import RecursiveChunker
from knowledge_agent.ingestion.pipeline import index_directory
from knowledge_agent.rag.generator import GenerationError, build_generator
from knowledge_agent.rag.pipeline import RAGError, answer_question
from knowledge_agent.retrieval.vector_store import DEFAULT_TOP_K, ChromaVectorStore, VectorStoreError

DEFAULT_ASK_SPACE = "nomic-768"


@dataclass(frozen=True)
class StoreHandle:
    store: ChromaVectorStore
    space: EmbeddingSpace
    note: str = ""
    fallback_used: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge-agent")
    commands = parser.add_subparsers(dest="command")

    index = commands.add_parser("index", help="Index every PDF in the documents directory")
    index.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks already stored instead of re-embedding them",
    )
    _add_space_flag(index)

    search = commands.add_parser("search", help="Search the indexed collection")
    search.add_argument("query", help="Natural language query")
    search.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="Results to return")
    search.add_argument("--document", default=None, help="Restrict to one document id")
    _add_space_flag(search)

    count = commands.add_parser("count", help="Print the number of stored chunks")
    _add_space_flag(count)

    commands.add_parser("spaces", help="List every embedding space and its stored chunks")

    compare = commands.add_parser(
        "compare", help="Run one query against every space and interleave the results"
    )
    compare.add_argument("query", help="Natural language query")
    compare.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="Results per space")

    ask = commands.add_parser(
        "ask", help="Answer a question from the indexed documents, with citations"
    )
    ask.add_argument("question", help="Question to answer")
    ask.add_argument(
        "-k", type=int, default=RAG_TOP_K, help="Chunks to retrieve as context"
    )
    ask.add_argument(
        "--min-distance",
        type=float,
        default=None,
        help="Skip generation and report no context when the best match is worse than this",
    )
    ask.add_argument("--json", action="store_true", help="Print the answer as JSON")
    _add_space_flag(ask, default=DEFAULT_ASK_SPACE, choices=list(SPACE_KEYS))

    serve = commands.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def _add_space_flag(
    parser: argparse.ArgumentParser, default: str | None = None, choices=None
) -> None:
    parser.add_argument(
        "--space",
        choices=list(SPACE_KEYS) if choices is None else choices,
        default=default,
        help="Pin an embedding space instead of auto-resolving (skip failover)",
    )


def build_store(args: argparse.Namespace, space: str | None = None) -> StoreHandle:
    preferred = space or getattr(args, "space", None)
    resolution = resolve_provider(preferred)

    return StoreHandle(
        store=ChromaVectorStore(provider=resolution.provider, space=resolution.space),
        space=resolution.space,
        note=resolution.note,
        fallback_used=resolution.fallback_used,
    )


def build_space_store(space: EmbeddingSpace) -> ChromaVectorStore:
    return ChromaVectorStore(space=space)


def _use_utf8_output() -> None:
    """Make printing arbitrary document text safe on a Windows console.

    The default cp1252 codec raises ``UnicodeEncodeError`` on characters that
    appear in maths-heavy papers, which would turn a successful search into a
    crash.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()

    args = build_parser().parse_args(argv)
    command = args.command or "index"

    if command == "index":
        return _index(args)
    if command == "search":
        return _search(args)
    if command == "count":
        return _count(args)
    if command == "spaces":
        return _spaces()
    if command == "compare":
        return _compare(args)
    if command == "ask":
        return _ask(args)
    if command == "serve":
        return _serve(args)

    return 1


def _announce(space: EmbeddingSpace, note: str) -> None:
    print(f"space={space.key} ({space.dimensions}-dim) collection={space.collection}")

    if note:
        print(f"note: {note}", file=sys.stderr)


def _index(args: argparse.Namespace) -> int:
    try:
        handle = build_store(args)
    except (EmbeddingConfigError, VectorStoreError) as exc:
        print(f"Cannot index: {exc}", file=sys.stderr)
        return 1

    _announce(handle.space, handle.note)

    report = index_directory(
        DOCUMENTS_DIR,
        handle.store,
        chunker=RecursiveChunker(),
        skip_existing=bool(getattr(args, "resume", False)),
    )

    report.space = handle.space.key
    report.collection = handle.space.collection
    report.fallback_used = handle.fallback_used
    report.space_reason = handle.note

    print(
        f"Indexed {report.chunks_indexed} chunks from {report.pages} pages "
        f"across {report.documents} documents"
    )

    if report.chunks_skipped:
        print(f"Skipped {report.chunks_skipped} already-stored chunks")

    print(f"Collection now holds {report.total_chunks} chunks")

    for failure in report.load_failures:
        print(f"Failed to load: {failure}", file=sys.stderr)

    for failure in report.store_failures:
        print(f"Failed to store: {failure}", file=sys.stderr)

    return 0 if report.ok else 1


def _search(args: argparse.Namespace) -> int:
    try:
        handle = build_store(args)
        results = handle.store.search(
            args.query,
            k=args.k,
            where={"document_id": args.document} if args.document else None,
        )
    except (EmbeddingConfigError, EmbeddingError, VectorStoreError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    _announce(handle.space, handle.note)

    if not results:
        print("No matching chunks")
        return 0

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. [{result.filename} p{result.page_number}] "
            f"distance={result.distance:.4f} id={result.chunk_id}"
        )
        print(f"   {result.content[:200]}")

    return 0


def _ask(args: argparse.Namespace) -> int:
    try:
        handle = build_store(args)
        generator = build_generator()
        answer = answer_question(
            args.question,
            handle.store,
            generator,
            space=handle.space,
            k=args.k,
            min_distance=args.min_distance,
        )
    except (
        EmbeddingConfigError,
        EmbeddingError,
        GenerationError,
        RAGError,
        VectorStoreError,
    ) as exc:
        print(f"Ask failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(answer.model_dump_json(indent=2))
        return 0

    _announce(handle.space, handle.note)
    print()
    print(answer.answer)
    print()

    if not answer.citations:
        print("Sources: none", file=sys.stderr)
        return 0

    print("Sources:")

    for citation in answer.citations:
        mark = "cited" if citation.cited else "unused"
        print(
            f"  [{citation.index}] {citation.filename} p{citation.page_number} "
            f"distance={citation.distance:.4f} {mark} id={citation.chunk_id}"
        )

    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("serve requires uvicorn: uv sync", file=sys.stderr)
        return 1

    print(f"API on http://{args.host}:{args.port} (POST /ask)")
    uvicorn.run("knowledge_agent.api:app", host=args.host, port=args.port)
    return 0


def _count(args: argparse.Namespace) -> int:
    collection, space = resolve_collection(getattr(args, "space", None))

    try:
        store = ChromaVectorStore(space=space, collection_name=collection)
        total = store.count()
    except VectorStoreError as exc:
        print(f"Count failed: {exc}", file=sys.stderr)
        return 1

    print(f"{space.key}: {total} chunks in {collection}")

    return 0


def _spaces() -> int:
    print(f"{'space':<14} {'model':<24} {'dim':>5}  {'collection':<26} chunks")

    for space in SPACES.values():
        try:
            total = build_space_store(space).count()
        except VectorStoreError as exc:
            print(f"{space.key:<14} {space.model:<24} {space.dimensions:>5}  "
                  f"{space.collection:<26} unavailable: {exc}", file=sys.stderr)
            return 1

        print(f"{space.key:<14} {space.model:<24} {space.dimensions:>5}  "
              f"{space.collection:<26} {total}")

    return 0


def _compare(args: argparse.Namespace) -> int:
    rows: list[tuple[float, str, int, object]] = []
    skipped: list[str] = []

    for space in SPACES.values():
        try:
            handle = build_store(args, space=space.key)
            results = handle.store.search(args.query, k=args.k)
        except (EmbeddingConfigError, EmbeddingError, VectorStoreError) as exc:
            skipped.append(f"{space.key}: {exc}")
            continue

        if not results:
            skipped.append(f"{space.key}: collection {space.collection} is empty")
            continue

        for rank, result in enumerate(results, start=1):
            rows.append((result.distance, space.key, rank, result))

    print(f"query: {args.query}")
    print("distance is not comparable across spaces; this shows what each retrieved.")
    print()

    for rank, (distance, key, _, result) in enumerate(sorted(rows), start=1):
        print(
            f"{rank:>3}  distance={distance:.4f}  space={key}  "
            f"[{result.filename} p{result.page_number}] id={result.chunk_id}"
        )
        print(f"     {result.content[:200]}")

    for entry in skipped:
        print(f"skipped {entry}", file=sys.stderr)

    return 0 if rows else 1