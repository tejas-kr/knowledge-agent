# AGENTS.md

RAG knowledge agent. Ingests PDFs from `documents/`, chunks them, embeds via Gemini, stores in ChromaDB.

## Commands

Managed with **uv** (not pip/poetry). Python 3.14.

```bash
uv run pytest                                    # 206 tests, ~20s
uv run pytest tests/test_chunker.py              # one file
uv run pytest tests/test_x.py::test_name         # one test
uv run pytest -k quota                           # by keyword
uv run knowledge-agent index                     # embed documents/ into ChromaDB
uv run knowledge-agent search "question" -k 5
uv run knowledge-agent count
```

- There is **no lint, format, or typecheck config** in this repo. Do not invent one or claim it runs.
- No CI. `.pre-commit-config.yaml` does not exist.
- Editing `pyproject.toml` (e.g. entry points) makes `uv run` rebuild/reinstall the package automatically.
- `uv run pytest` must run from the repo root — there is no configured `testpaths`.

## Pipeline

`documents/*.pdf` → `ingestion/loader.py` → `ingestion/chunker.py` → `embeddings/provider.py` → `retrieval/vector_store.py`

| Path | Role |
|---|---|
| `ingestion/loader.py` | `Loader` ABC + `PdfLoader` → `DocumentPage` |
| `ingestion/chunker.py` | `Chunker` ABC + `RecursiveChunker` → `Chunk` |
| `ingestion/pipeline.py` | `load_documents`, `index_directory`, `IngestReport` |
| `embeddings/provider.py` | `EmbeddingProvider` ABC + `GeminiEmbeddingProvider` |
| `retrieval/vector_store.py` | `VectorStore` ABC + `ChromaVectorStore` |
| `schemas/` | pydantic models: `DocumentPage`, `Chunk`, `SearchResult` — pure data |
| `main.py` | CLI only; no business logic |
| `config.py` | paths + `.env` loading (runs at import) |

Each module keeps its own exception hierarchy rooted in a per-module base (`LoaderError`, `ChunkerError`, `EmbeddingError`, `VectorStoreError`) so callers can catch one type.

## Gotchas that cost real debugging time

1. **Do not re-export in `__init__.py` — they are empty on purpose.** The console script is `knowledge_agent.main:main`. Adding `from .main import main` to `__init__.py` makes `knowledge_agent.main` resolve to the *function*, shadowing the submodule, and breaks `patch("knowledge_agent.main.X")`. Entry point spec must be `module:attr`.

2. **Gemini free-tier quotas are tight** (`gemini-embedding-001`): 100 `embed_content` requests/minute, **1000/day**. Batching 100 texts per request is the hard cap. `index_directory` batches chunks *across pages* — a per-page call means 499 requests for this corpus and will hit the limit. Treat re-indexing as quota-expensive.

3. **Embedding dimension is 3072, not 768.** That is the `gemini-embedding-001` default. A Chroma collection is permanently bound to the dimension it was created with, so switching dimensions means deleting `data/chroma` and re-indexing from scratch. `output_dimensionality=768` is supported if you want the smaller vector.

4. **ChromaDB holds file handles on Windows.** You cannot delete or overwrite `data/chroma` while a store is alive in the process; do it from a fresh process.

5. **Use `import pymupdf`, never `import fitz`.** `fitz` is deprecated and prints a warning on every run.

6. `document_id` is `path.stem`, so it keeps dots (e.g. `1706.03762v7`). That is fine as a Chroma id — don't "sanitise" it and break the existing ids.

## Current state

- `documents/` holds 13 PDFs / 499 pages → ~2511 chunks at default `chunk_size=1000`.
- **`data/chroma` is partially indexed: 819 of 2511 chunks.** The daily quota was exhausted mid-run. Re-run `uv run knowledge-agent index` to finish; `upsert` on `document_id:page_number:chunk_index` makes it idempotent, but it re-embeds everything it walks, so it spends quota on already-stored chunks.
- `data/` is gitignored.

## Testing conventions

- **The suite makes no network calls.** Inject fakes: `FakeClient` (`tests/test_provider.py`), `StubProvider` (`tests/test_vector_store.py`), `FakeStore` (`tests/test_pipeline.py`). `GeminiEmbeddingProvider` takes an optional `client=`, and `ChromaVectorStore` takes an optional `client=` / `directory=` — use those seams instead of mocking module internals.
- Retry/backoff tests inject a fake `sleep` callable; never call `time.sleep` for real.
- `tests/test_persistence.py` spawns **real subprocesses** (index in one process, query in another) on purpose — that is what makes it a genuine restart/persistence test. It is the slowest file.
- Integration tests use real ChromaDB in `tmp_path`; they are hermetic.
- `FakeStore` in `test_pipeline.py` models upsert semantics (same id does not grow the store) so re-index tests mean something.
- When changing batch sizes, remember a store failure is now recorded **once per batch**, not per page.
- `DeprecationWarning`s from `google/genai` and `chromadb` are upstream. Not ours; don't chase them.

## Conventions

- 4-space indent, double quotes, full type hints, no comments in source.
- Wrap third-party failures: `raise OurError(...) from exc`. Preserve the cause.
- Public behaviour is covered by tests that assert the exact values handed to the boundary (ids, metadata, task types, batch boundaries) rather than only the happy-path return.
- Distance, not score: Chroma returns cosine distance where **lower is more similar**. `SearchResult.similarity` is the `1 - distance` convenience view.
