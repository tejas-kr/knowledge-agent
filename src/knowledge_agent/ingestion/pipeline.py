from dataclasses import dataclass, field
from pathlib import Path

from knowledge_agent.ingestion.chunker import Chunker, RecursiveChunker
from knowledge_agent.ingestion.loader import DocumentNotFoundError, LoaderError, PdfLoader
from knowledge_agent.retrieval.vector_store import VectorStore, VectorStoreError
from knowledge_agent.schemas.document import DocumentPage


DEFAULT_STORE_BATCH_SIZE = 100


@dataclass
class IngestReport:
    documents: int = 0
    pages: int = 0
    chunks_indexed: int = 0
    total_chunks: int = 0
    load_failures: list[LoaderError] = field(default_factory=list)
    store_failures: list[VectorStoreError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.load_failures and not self.store_failures


def load_documents(directory: Path) -> tuple[list[DocumentPage], list[LoaderError]]:
    loader = PdfLoader()

    if not directory.is_dir():
        raise DocumentNotFoundError(f"Directory not found: {directory}")

    paths = sorted(p for p in directory.iterdir() if p.is_file() and loader.supports(p))

    pages: list[DocumentPage] = []
    failures: list[LoaderError] = []

    for path in paths:
        try:
            pages.extend(loader.load(path))
        except LoaderError as exc:
            failures.append(exc)

    return pages, failures


def index_directory(
    directory: Path,
    store: VectorStore,
    chunker: Chunker | None = None,
    batch_size: int = DEFAULT_STORE_BATCH_SIZE,
) -> IngestReport:
    chunker = chunker or RecursiveChunker()

    pages, load_failures = load_documents(directory)

    report = IngestReport(
        documents=len({page.filename for page in pages}),
        pages=len(pages),
        load_failures=load_failures,
    )

    pending: list = []

    for page in pages:
        pending.extend(chunker.chunk(page))

        if len(pending) >= batch_size:
            _flush(pending, store, report)
            pending = []

    if pending:
        _flush(pending, store, report)

    report.total_chunks = store.count()

    return report


def _flush(pending: list, store: VectorStore, report: IngestReport) -> None:
    try:
        report.chunks_indexed += len(store.add_chunks(pending))
    except VectorStoreError as exc:
        report.store_failures.append(exc)

    pending.clear()
