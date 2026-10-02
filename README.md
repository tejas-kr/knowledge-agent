# knowledge-agent

A local RAG agent over a folder of research PDFs. Ingests `documents/*.pdf`, chunks them,
embeds each chunk, stores vectors in ChromaDB, and answers questions with a citation list
pointing back at the exact page.

Everything runs on your machine. The default answer model is a local Ollama model, so no
API key is needed and there are no quota limits.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

```bash
uv sync
ollama pull nomic-embed-text-v2-moe   # embeddings, 957 MB
ollama pull qwen3:1.7b                # answers, 1.1 GB
```

Both models were chosen to fit a 4 GB card. If you have more VRAM, larger models work fine —
see [Configuration](#configuration).

## Index the documents

```bash
uv run knowledge-agent index
```

This walks `documents/*.pdf`, chunks, embeds, and stores everything in `data/chroma`.
Takes about 2 minutes for 13 papers / 499 pages.

Re-running re-embeds every chunk, so prefer `--resume` when topping up a partial index:

```bash
uv run knowledge-agent index --resume    # only embed what is missing
```

Check what you have:

```bash
uv run knowledge-agent count
uv run knowledge-agent spaces
```

```
space          model                      dim  collection                 chunks
gemini-3072    gemini-embedding-001      3072  knowledge                  2205
nomic-768      nomic-embed-text-v2-moe    768  knowledge--nomic-768       2511
```

## Ask a question

```bash
uv run knowledge-agent ask "What is multi-head attention?"
```

```
Multi-head attention is a mechanism used in the Transformer model to jointly
attend to information from different representation subspaces...

Sources:
  [1] 1706.03762v7.pdf p5 distance=0.4666 cited id=1706.03762v7:5:0
  [2] 2303.18223v19.pdf p25 distance=0.4919 unused id=2303.18223v19:25:4
```

The source list comes from the vector store, not from the model's prose, so the filenames
and page numbers are always real. `cited` means the answer actually referenced that chunk
with a `[n]` marker; `unused` means it was retrieved but the model leaned elsewhere.

Expect **8-20 seconds per answer** on a 4 GB card. The model is small, and it is not fast.

### Useful flags

| Flag | What it does |
|---|---|
| `-k N` | How many chunks to retrieve as context (default 5) |
| `--min-distance F` | Skip generation entirely when the best match is worse than `F` |
| `--json` | Machine-readable output |
| `--space` | Pin `gemini-3072` or `nomic-768` |

```bash
# JSON for scripting
uv run knowledge-agent ask "What is multi-head attention?" -k 2 --json

# refuse to answer on a weak match, without paying for generation
uv run knowledge-agent ask "Population of Lisbon in 1843?" --min-distance 0.45
```

`--min-distance` is the honest guard against confident nonsense. Cosine distance is
**lower is better**; a weak match is a high number. When the best hit is worse than your
threshold, you get "no context" and nothing is generated.

## Search without generating

Faster, and useful when you just want to see what was retrieved:

```bash
uv run knowledge-agent search "multi-head attention" -k 5
uv run knowledge-agent search "BLEU scores" --document 1706.03762v7
```

`compare` runs one query against every space and interleaves the results, which is how you
compare two embedding models directly:

```bash
uv run knowledge-agent compare "what is attention?" -k 3
```

## Run the API

```bash
uv run knowledge-agent serve --port 8000
```

```
API on http://127.0.0.1:8000 (POST /ask)
```

Then in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?", "k": 2}'
```

```json
{
  "question": "What is multi-head attention?",
  "answer": "Multi-head attention is a mechanism used in the Transformer...",
  "citations": [
    {
      "index": 1,
      "chunk_id": "1706.03762v7:5:0",
      "document_id": "1706.03762v7",
      "filename": "1706.03762v7.pdf",
      "page_number": 5,
      "distance": 0.4665,
      "cited": true
    }
  ],
  "space": "nomic-768",
  "top_distance": 0.4665,
  "context_chunks": 2
}
```

Health check, for monitoring:

```bash
curl http://127.0.0.1:8000/health     # {"status":"ok"}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `question` | string | required | blank returns 422 |
| `k` | int | 5 | chunks to retrieve |
| `space` | string | `nomic-768` | `gemini-3072` or `nomic-768` |
| `min_distance` | float | none | see above |

Status codes: `200` ok (including a legitimate "no context" answer), `422` bad question,
`502` generation failed, `503` could not reach Ollama or a store would not open.

Interactive docs are at `http://127.0.0.1:8000/docs` if you have `uvicorn` installed.

For development with auto-reload:

```bash
uv run uvicorn knowledge_agent.api:app --reload
```

The server holds one store per space and one generator for its whole lifetime, because
Chroma keeps file handles open on Windows. Startup is lazy, so the API key and Ollama are
only needed once you send a real question.

## Configuration

Copy the example and edit as needed:

```bash
cp .example.env .env
```

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | embedding model |
| `OLLAMA_GENERATION_MODEL` | `qwen3:1.7b` | answer model |
| `OLLAMA_GENERATION_NUM_CTX` | `8192` | context window for answering |
| `GENERATOR_BACKEND` | `ollama` | `ollama` or `gemini` |
| `RAG_TOP_K` | `5` | default context chunks |
| `RAG_MAX_CONTEXT_CHARS` | `12000` | context budget before trimming |
| `RAG_CITE_INSTRUCTION` | `1` | append the citation-marker ask |
| `GEMINI_API_KEY` | — | only for `GENERATOR_BACKEND=gemini` |

Two settings worth understanding:

**`OLLAMA_GENERATION_NUM_CTX` is 8192, not Ollama's 4096 default.** A full RAG prompt is
3-4k tokens, so the default can silently truncate the context block. 8192 costs roughly
900 MB of KV cache for `qwen3:1.7b`; drop to 4096 if VRAM is tight.

**`RAG_CITE_INSTRUCTION=1` appends a short instruction asking for `[n]` markers.** Without
it, small local models answer correctly but cite nothing, and every citation reports
`cited=false`. Set it to `0` to turn it off. The system prompt itself is never rewritten.

### Picking a different model

The generation model is independent of the embedding model, so you can mix them. Check VRAM
headroom before switching: a model that does not fit in VRAM falls back to system RAM and
gets dramatically slower.

| Model | Size | Notes |
|---|---|---|
| `qwen3:1.7b` | 1.1 GB | default, best quality per GB on a 4 GB card |
| `llama3.2:3b` | ~2 GB | fits, stronger, slower |
| `gemma3:1b` | ~800 MB | fastest, weaker synthesis |
| `qwen3:4b` | ~2.5 GB | fits, but little room for KV cache |

```bash
ollama pull llama3:1.7b
# then set OLLAMA_GENERATION_MODEL=llama3:1.7b in .env
```

To answer with Gemini instead (slower to answer, no local VRAM cost, but a free-tier quota
and occasional `503`s):

```bash
# in .env
GENERATOR_BACKEND=gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.8-flash
```

## Two embedding spaces

Two models are configured, each with its own collection so they never mix vectors:

| Space | Model | Dims | Collection | Chunks |
|---|---|---|---|---|
| `nomic-768` | `nomic-embed-text-v2-moe` | 768 | `knowledge--nomic-768` | 2511 |
| `gemini-3072` | `gemini-embedding-001` | 3072 | `knowledge` | 2205 |

`nomic-768` is complete and is the default everywhere. `gemini-3072` is 2205/2511, short
because the Gemini free tier caps at 1000 embedding requests per day. Top it up on another
day with `index --resume --space gemini-3072`.

A collection is permanently bound to the vector width it was created with, so changing a
space's dimensions means a new collection and a full re-index, not a re-query.

Any command takes `--space` to pin one explicitly. Pinning skips the health probe and
failover on purpose, so a pinned run reports that space's own failure instead of quietly
answering from a different model.

## Tests

```bash
uv run pytest              # 492 tests
uv run pytest -k citation  # by keyword
uv run pytest tests/test_api.py
```

The suite makes no network calls. It uses injected fakes rather than mocking internals, and
runs in about 45 seconds.

## How it works

```
documents/*.pdf
  -> ingestion/loader.py     extract text per page
  -> ingestion/chunker.py    split into overlapping chunks
  -> embeddings/             embed, one space per collection
  -> retrieval/              ChromaDB search, cosine distance
  -> rag/prompt.py           numbered context, one-pass prompt render
  -> rag/generator.py        Ollama or Gemini, temperature 0
  -> rag/pipeline.py         assemble answer + citations
```

`data/` is gitignored and holds the Chroma database and cooldown state. Deleting it means
re-indexing.

## Known limitations

**Retrieval is the weak link, not generation.** A 1.7B model over nomic retrieval will
sometimes answer confidently from the wrong document while citing consistently. The
citations tell you *which chunks* were used; they do not guarantee the answer used them
faithfully. Use `--min-distance` to refuse low-confidence matches.

**Answers take 8-20 seconds.** Expected on a 4 GB card.

**The citation markers are best-effort.** `cited` is true only when the model wrote a `[n]`.
It is not a grounding guarantee.

**`gemini-3072` is incomplete** and will stay that way until topped up across several days.
