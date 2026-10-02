from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Answer pipeline (rag/)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

# Which generator answers questions: "ollama" or "gemini".
# Local generation is the default because the Gemini free tier answers with
# transient 503 UNAVAILABLE under load, and needs no quota.
GENERATOR_BACKEND = os.getenv("GENERATOR_BACKEND", "ollama")

# Local generation model. 1.7B Q4 is ~1.1 GB, which leaves room for the KV
# cache on a 4 GB card; the 8B qwen3-embedding lesson applies here too.
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "qwen3:1.7b")

# Ollama defaults to a 4096-token window, but a full RAG prompt is ~3-4k
# tokens, so the default can truncate the context block. 8192 costs roughly
# 900 MB of KV cache for this model; drop to 4096 to halve that on a tight card.
OLLAMA_GENERATION_NUM_CTX = int(os.getenv("OLLAMA_GENERATION_NUM_CTX", "8192"))

# Appended to the question so the model emits [1]-style markers. On by default:
# the small local models do not infer citation markers, and without it every
# citation reports cited=False. Set to 0 to send the question verbatim.
RAG_CITE_INSTRUCTION = os.getenv("RAG_CITE_INSTRUCTION", "1") not in {
    "0",
    "false",
    "no",
}
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL"
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE")
OLLAMA_TIMEOUT = os.getenv("OLLAMA_TIMEOUT")
EMBEDDING_FALLBACK_ORDER = os.getenv("EMBEDDING_FALLBACK_ORDER")
GEMINI_QUOTA_RESET_HOURS = os.getenv("GEMINI_QUOTA_RESET_HOURS")
PROVIDER_HEALTH_FILE = os.getenv("PROVIDER_HEALTH_FILE")