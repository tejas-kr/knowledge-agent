import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

CHILD = '''
import sys
from pathlib import Path

from knowledge_agent.embeddings.provider import EmbeddingProvider
from knowledge_agent.ingestion.chunker import RecursiveChunker
from knowledge_agent.ingestion.pipeline import index_directory
from knowledge_agent.retrieval.vector_store import ChromaVectorStore

VOCABULARY = "abcdefghijklmnopqrstuvwxyz"


class StubProvider(EmbeddingProvider):
    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        lowered = text.lower()
        return [float(lowered.count(letter)) for letter in VOCABULARY]


def main():
    mode = sys.argv[1]
    documents = Path(sys.argv[2])
    chroma = Path(sys.argv[3])

    store = ChromaVectorStore(
        provider=StubProvider(), directory=chroma, collection_name="persist"
    )

    if mode == "index":
        report = index_directory(
            documents, store, chunker=RecursiveChunker(chunk_size=60, chunk_overlap=10)
        )
        print(f"chunks={report.chunks_indexed} total={report.total_chunks} ok={report.ok}")
        return 0 if report.ok else 1

    if mode == "query":
        results = store.search(sys.argv[4], k=3)
        print(f"count={store.count()}")
        for result in results:
            print(f"{result.chunk_id}|{result.distance:.6f}|{result.content.strip()}")
        return 0

    return 2


sys.exit(main())
'''


def run_child(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def child(tmp_path: Path) -> Path:
    path = tmp_path / "_child.py"
    path.write_text(CHILD, encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    documents = tmp_path / "docs"
    documents.mkdir()
    for name, texts in {
        "attention.pdf": ["Transformers use attention mechanisms", "attention is all you need"],
        "retrieval.pdf": ["Vector databases store embeddings", "augmented generation pipelines"],
    }.items():
        doc = pymupdf.open()
        for text in texts:
            doc.new_page().insert_text((72, 72), text)
        doc.save(documents / name)
        doc.close()
    return documents


def test_vectors_survive_a_process_restart(child, corpus, tmp_path):
    chroma = tmp_path / "chroma"

    indexed = run_child(child, "index", str(corpus), str(chroma))

    assert indexed.returncode == 0, indexed.stderr
    assert "chunks=4" in indexed.stdout

    queried = run_child(child, "query", str(corpus), str(chroma), "embeddings")

    assert queried.returncode == 0, queried.stderr
    assert "count=4" in queried.stdout

    lines = [line for line in queried.stdout.splitlines() if "|" in line]
    assert lines, queried.stdout

    top_id, top_distance, top_content = lines[0].split("|", 2)
    assert top_content == "Vector databases store embeddings"
    assert top_id.startswith("retrieval:")

    distances = [float(line.split("|", 2)[1]) for line in lines]
    assert distances == sorted(distances)


def test_reindexing_in_a_new_process_is_idempotent(child, corpus, tmp_path):
    chroma = tmp_path / "chroma"

    first = run_child(child, "index", str(corpus), str(chroma))
    second = run_child(child, "index", str(corpus), str(chroma))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "total=4" in second.stdout


def test_query_ordering_survives_restart(child, corpus, tmp_path):
    chroma = tmp_path / "chroma"

    run_child(child, "index", str(corpus), str(chroma))
    queried = run_child(child, "query", str(corpus), str(chroma), "attention")

    lines = [line for line in queried.stdout.splitlines() if "|" in line]
    assert lines
    assert "attention" in lines[0].split("|", 2)[2].lower()


def test_chroma_files_are_written_to_disk(child, corpus, tmp_path):
    chroma = tmp_path / "chroma"

    run_child(child, "index", str(corpus), str(chroma))

    assert chroma.is_dir()
    assert any(chroma.rglob("*.bin"))
    assert any(chroma.rglob("*.sqlite3"))


def test_query_against_empty_store_in_a_new_process(child, tmp_path):
    chroma = tmp_path / "chroma-empty"
    documents = tmp_path / "empty"
    documents.mkdir()

    queried = run_child(child, "query", str(documents), str(chroma), "anything")

    assert queried.returncode == 0, queried.stderr
    assert "count=0" in queried.stdout
