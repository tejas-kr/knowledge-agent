import argparse
import sys

from knowledge_agent.config import DOCUMENTS_DIR
from knowledge_agent.embeddings.provider import GeminiEmbeddingProvider
from knowledge_agent.ingestion.chunker import RecursiveChunker
from knowledge_agent.ingestion.pipeline import index_directory
from knowledge_agent.retrieval.vector_store import DEFAULT_TOP_K, ChromaVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge-agent")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("index", help="Index every PDF in the documents directory")

    search = commands.add_parser("search", help="Search the indexed collection")
    search.add_argument("query", help="Natural language query")
    search.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="Results to return")
    search.add_argument("--document", default=None, help="Restrict to one document id")

    commands.add_parser("count", help="Print the number of stored chunks")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "index"

    if command == "index":
        return _index()
    if command == "search":
        return _search(args)
    if command == "count":
        return _count()

    return 1


def _index() -> int:
    store = ChromaVectorStore(provider=GeminiEmbeddingProvider())

    report = index_directory(DOCUMENTS_DIR, store, chunker=RecursiveChunker())

    print(
        f"Indexed {report.chunks_indexed} chunks from {report.pages} pages "
        f"across {report.documents} documents"
    )
    print(f"Collection now holds {report.total_chunks} chunks")

    for failure in report.load_failures:
        print(f"Failed to load: {failure}", file=sys.stderr)

    for failure in report.store_failures:
        print(f"Failed to store: {failure}", file=sys.stderr)

    return 0 if report.ok else 1


def _search(args: argparse.Namespace) -> int:
    store = ChromaVectorStore(provider=GeminiEmbeddingProvider())

    where = {"document_id": args.document} if args.document else None

    results = store.search(args.query, k=args.k, where=where)

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


def _count() -> int:
    store = ChromaVectorStore(provider=GeminiEmbeddingProvider())

    print(f"{store.count()} chunks")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
