# AGENTS.md

RAG knowledge agent. Ingests PDFs from `documents/`, chunks them, embeds via Gemini, stores in ChromaDB.

## Commands

Managed with **uv** (not pip/poetry). Python 3.14.

```bash
uv run pytest                                    # 450 tests, ~21s
uv run pytest tests/test_chunker.py              # one file
uv run pytest tests/test_x.py::test_name         # one test
uv run pytest -k quota                           # by keyword
uv run knowledge-agent index                     # embed documents/ into ChromaDB
uv run knowledge-agent index --resume            # ONLY embed chunks not already stored
uv run knowledge-agent search "question" -k 5
uv run knowledge-agent count
uv run knowledge-agent spaces                    # list spaces and their chunk counts
uv run knowledge-agent compare "question" -k 5   # run one query against every space
uv run knowledge-agent ask "question"            # grounded answer + source list
uv run knowledge-agent ask "question" --json     # same, machine-readable
uv run knowledge-agent serve --port 8000         # POST /ask over HTTP
```

- Any of `index`, `search`, `count`, `ask` take `--space gemini-3072` or `--space nomic-768`.
- `ask` also takes `-k` (context chunks) and `--min-distance` (skip generation and report no context when the best match is worse than this).
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
| `embeddings/space.py` | `EmbeddingSpace` + the `SPACES` registry (the two collections) |
| `embeddings/provider.py` | `EmbeddingProvider` ABC (incl. `space`) + `GeminiEmbeddingProvider` |
| `embeddings/ollama.py` | `OllamaEmbeddingProvider` over `/api/embed` |
| `embeddings/health.py` | `ProviderHealth` cooldown records + quota-reset parsing |
| `embeddings/resolver.py` | `resolve_provider` (probe, failover, latch) + `resolve_collection` |
| `retrieval/vector_store.py` | `VectorStore` ABC + `ChromaVectorStore` |
| `rag/prompt.py` | `SYSTEM_PROMPT` + `render_prompt` / `format_context` / `trim_context` / `cited_indexes` |
| `rag/generator.py` | `Generator` ABC + `GeminiGenerator` + `build_generator` (`GENERATOR_BACKEND`) |
| `rag/ollama.py` | `OllamaGenerator` over `/api/chat` + `strip_thinking` |
| `rag/pipeline.py` | `answer_question` — the only place retrieval meets generation |
| `schemas/` | pydantic models: `DocumentPage`, `Chunk`, `SearchResult`, `Citation`, `RAGAnswer` — pure data |
| `api.py` | FastAPI `app`: `POST /ask`, `GET /health`, per-space store cache |
| `main.py` | CLI only; no business logic. `build_store` is the test seam for wiring |
| `config.py` | paths + `.env` loading (runs at import) |

Each module keeps its own exception hierarchy rooted in a per-module base (`LoaderError`, `ChunkerError`, `EmbeddingError`, `VectorStoreError`, `GenerationError`, `RAGError`) so callers can catch one type.

## Answers and citations

`ask` and `POST /ask` run `rag/pipeline.py`: `store.search` → `format_context` → `render_prompt` → `Generator.generate` → `build_citations`.

- **The prompt is substituted with a regex, never `str.format`.** The corpus is arXiv PDFs full of LaTeX, so `format_context` output routinely contains `{` and `}`. `str.format` raises `KeyError` on those, and even with escaping it would rescan inserted text and re-expand a chunk that literally contains `{question}`. `render_prompt` substitutes `{context}` and `{question}` in one pass and never looks at what it inserted.
- **Provenance comes from `SearchResult`, never from generated text.** The context block is labelled `[n] (filename, page N)` so the model can attribute its own claims, but `Citation.filename` / `page_number` / `distance` are read off the retrieved `SearchResult`. A model that invents `[9]` cannot invent a citation.
- **Every retrieved chunk is cited; `cited` is just a flag.** `cited_indexes` reads `[1]`, `[1,2]`, `[1, 2]`. A model that ignores the markers still loses no provenance, which is why there is no heuristic `grounded` flag — `top_distance` is published instead and the caller decides.
- **The citation ask lives outside `SYSTEM_PROMPT`.** `CITATION_INSTRUCTION` is appended to the *question*, because the small local models do not infer `[n]` markers: `qwen3:1.7b` returned every citation as `cited=False` until the instruction was added. Do not "tidy" it into the system prompt — that template is the one the user specified verbatim, and a test asserts it is unchanged.
- **`OllamaGenerator` posts to `/api/chat`, not `/api/generate`.** An instruct model only applies its chat template on the chat endpoint, and the prompt already carries the system instructions.
- **`num_ctx` is 8192, not Ollama's 4096 default.** A full RAG prompt is ~3-4k tokens, so the default can silently truncate the context block. 8192 costs roughly 900 MB of KV cache for `qwen3:1.7b`; 4096 halves it.
- **`strip_thinking` is not optional.** Qwen3 emits `<thinking>` blocks; reasoning text in `answer` would put stray `[n]` markers into `cited_indexes` and credit sources the answer never used. An *unclosed* block discards the remainder and surfaces as "no text", because a truncated reasoning block means there is no answer to trust.
- **Local generation is the default, Gemini is reachable but not default.** `GENERATOR_BACKEND` picks. A 1.7B model answering in 13-43s is more reliable than a free-tier endpoint returning `503 UNAVAILABLE` under load.
- **Over-budget context drops the lowest-ranked chunks**, never truncates mid-sentence, and always keeps at least one.
- `ask` defaults to `nomic-768` (the complete index); `--space gemini-3072` opts into the partial one.
- The **generator** defaults to local `qwen3:1.7b` (~1.1 GB Q4, fits a 4 GB card with room for KV cache). The generation model is independent of the embedding space: `ask` can embed with nomic and answer with Ollama or Gemini.
- `api.py` caches one store per space with `lru_cache` and one generator, because Chroma holds file handles on Windows and `genai.Client` owns a connection pool. Both stay lazy so importing `api` needs neither a Chroma directory nor an API key.

## Embedding spaces

Two independently trained models share one Chroma database but **never** one collection.

| Space key | Model | Dims | Collection |
|---|---|---|---|
| `gemini-3072` | `gemini-embedding-001` | 3072 | `knowledge` |
| `nomic-768` | `nomic-embed-text-v2-moe` (Ollama) | 768 | `knowledge--nomic-768` |

nomic-embed-text is trained with task prefixes. `EmbeddingSpace` carries both
`query_instruction` (`search_query: `) and `document_instruction`
(`search_document: `), and the provider applies the right one per call. Measured
with the prefixes, a relevant document scores 0.38-0.52 against an irrelevant
one at 0.01-0.07. Never "tidy" the prefixes away — nomic degrades badly without them.

## Gotchas that cost real debugging time

1. **Do not re-export in `__init__.py` — they are empty on purpose.** The console script is `knowledge_agent.main:main`. Adding `from .main import main` to `__init__.py` makes `knowledge_agent.main` resolve to the *function*, shadowing the submodule, and breaks `patch("knowledge_agent.main.X")`. Entry point spec must be `module:attr`.

2. **A collection is guarded by `embedding_space` identity, not by width.** `ChromaVectorStore._verify_space` reads the `embedding_space` collection metadata and refuses a mismatch. Width happens to differ today (3072 vs 768), but it is a coincidence of current config, not an invariant — and width alone was measured to be useless: two 768-dim models put the *correct* document at 0.02-0.09 cosine versus 0.26-0.45 within one model. A collection created before that metadata existed (`knowledge`) is checked against `LEGACY_COLLECTIONS`, which records which space owns it. Never weaken the guard to a dimension comparison.

3. **Gemini free-tier quotas are tight** (`gemini-embedding-001`): 100 `embed_content` requests/minute, **1000/day**. Batching 100 texts per request is the hard cap. `index_directory` batches chunks *across pages* — a per-page call means 499 requests for this corpus and will hit the limit. Treat re-indexing as quota-expensive.

4. **Gemini's dimension default is 3072.** That is the `gemini-embedding-001` default. A Chroma collection is permanently bound to the vectors it was created with, so switching a space's width means a new collection, not a re-index of the old one. `output_dimensionality=768` is supported if you want the smaller vector.

5. **Failover resolves *before* the store is built, and the resolution never changes mid-run.** A run indexes into exactly one collection, so switching providers halfway through would write the wrong space's vectors into it. `resolve_provider` probes each candidate in `SPACES` order, and `LatchingProvider` makes an exhausted provider cost exactly one request instead of one per batch (`index_directory` records a store failure once per batch and keeps going). **Pinned runs are latched too** — a pin chooses *which* space, it does not licence retrying a dead provider 26 times. The next run picks the fallback because the cooldown is on disk.

6. **Cooldown persistence is best-effort and must never abort a run.** `data/provider_health.json` is untrusted: a corrupt, truncated, or unwritable file degrades to "no health information". Parsing a reset hint from a quota message is best-effort too — Gemini does not reliably say when the daily window rolls over, so `GEMINI_QUOTA_RESET_HOURS` is the floor.

7. **`--space` is a pin, not a preference.** An explicit space skips the probe, skips cooldown, and does not fail over. That is deliberate: a pinned run must report the pinned space's own failure instead of quietly answering from a different model.

8. **Pick an Ollama model that fits in VRAM.** This box has a GTX 1650 with 4GB. `qwen3-embedding` (8B, 4.7GB) does not fit, so it runs 100% on CPU at 2.9s/chunk — a full index did not finish overnight. `nomic-embed-text-v2-moe` is 957MB and hits the GPU at 0.055s/chunk, about 50x faster, and indexes the whole corpus in ~2 minutes. Check VRAM headroom before switching models. The default `OLLAMA_TIMEOUT` is 600s regardless; a timeout is a throughput failure, not a quota signal.

8. **ChromaDB holds file handles on Windows.** You cannot delete or overwrite `data/chroma` while a store is alive in the process; do it from a fresh process.

9. **Use `import pymupdf`, never `import fitz`.** `fitz` is deprecated and prints a warning on every run.

10. `document_id` is `path.stem`, so it keeps dots (e.g. `1706.03762v7`). That is fine as a Chroma id — don't "sanitise" it and break the existing ids.

11. **The chunk id is owned by the schema.** `Chunk.chunk_id` is the single source of truth for `document_id:page_number:chunk_index`; the pipeline and store both read it. There is no `ChromaVectorStore.chunk_id` static method — don't reintroduce a second implementation.

## Current state

- `documents/` holds 13 PDFs / 499 pages → ~2511 chunks at default `chunk_size=1000`.
- **`data/chroma` `knowledge` (gemini-3072) holds 2205 of 2511 chunks.** 306 remain, blocked on the Gemini free-tier *daily* cap (1000 requests/day), not on code.
- **`data/chroma` `knowledge--nomic-768` (nomic-768) is empty.** `nomic-embed-text-v2-moe` is pulled locally and reaches 768 dims; it is the drop-in target for the remaining chunks, and indexing the full corpus takes about 2 minutes.
- Without `--resume`, `index` re-embeds every chunk it walks even though `upsert` makes it idempotent. Prefer `--resume` when topping up a partial index.
- A store failure is recorded **once per batch** and does not abort the run, so one exhausted quota run still stores every batch that succeeded. Exit code is `1` if any batch failed.
- `data/` is gitignored.

## Testing conventions

- **The suite makes no network calls.** Inject fakes: `FakeClient` (`tests/test_provider.py`), `StubProvider` (`tests/test_vector_store.py`), `FakeStore` (`tests/test_pipeline.py`), `FakeHTTP` (`tests/test_ollama.py`), `FakeProvider` (`tests/test_resolver.py`). `GeminiEmbeddingProvider` takes an optional `client=`, `OllamaEmbeddingProvider` takes an optional `client=`, and `ChromaVectorStore` takes an optional `client=` / `directory=` — use those seams instead of mocking module internals.
- **Every `EmbeddingProvider` double must implement `space`.** It is an abstract property, and the store derives its collection name from it, so a double without one cannot even be instantiated. A test double also has to pass a matching `collection_name` when it is deliberately aiming at a legacy collection, or the store silently opens the double's *own* collection instead.
- `resolve_provider` takes `builder=`, `health=`, and `order=` precisely so the suite can resolve providers with no network and no disk.
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
