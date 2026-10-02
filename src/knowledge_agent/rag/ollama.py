import re

import httpx

from knowledge_agent.config import (
    OLLAMA_GENERATION_MODEL,
    OLLAMA_GENERATION_NUM_CTX,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
)
from knowledge_agent.rag.generator import (
    GenerationAPIError,
    GenerationConfigError,
    GenerationError,
    Generator,
    GROUNDED_TEMPERATURE,
)

CHAT_PATH = "/api/chat"
DEFAULT_OLLAMA_TIMEOUT = 600.0
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"

_THINKING_BLOCK = re.compile(
    r"<(think|thinking)>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_UNCLOSED_THINKING = re.compile(r"<(think|thinking)>.*", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop Qwen3's reasoning blocks from the answer.

    Ollama honours ``think: false``, but older servers ignore it and some
    templates leak the block anyway. Reasoning text would otherwise land in
    ``RAGAnswer.answer`` and break ``cited_indexes`` — a stray ``[1]`` inside a
    deliberation would credit a source the answer never used.
    """
    return _UNCLOSED_THINKING.sub("", _THINKING_BLOCK.sub("", text)).strip()


class OllamaGenerator(Generator):
    """Answer generation over a local Ollama model.

    Uses ``/api/chat`` rather than ``/api/generate``: the prompt already
    contains the system instructions, but an instruct model like Qwen3 only
    applies its chat template on the chat endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = GROUNDED_TEMPERATURE,
        num_ctx: int | None = None,
        timeout: float | None = None,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_model = model or OLLAMA_GENERATION_MODEL

        if not resolved_model.strip():
            raise GenerationConfigError("model cannot be empty")

        resolved_host = (host or OLLAMA_HOST or "").strip()

        if not resolved_host:
            raise GenerationConfigError(
                "No Ollama host: set OLLAMA_HOST or pass host=..."
            )

        if temperature < 0:
            raise GenerationConfigError(
                f"temperature cannot be negative, got {temperature}"
            )

        resolved_num_ctx = (
            num_ctx if num_ctx is not None else OLLAMA_GENERATION_NUM_CTX
        )

        if resolved_num_ctx <= 0:
            raise GenerationConfigError(
                f"num_ctx must be positive, got {resolved_num_ctx}"
            )

        if not keep_alive.strip():
            raise GenerationConfigError("keep_alive cannot be empty")

        self.model = resolved_model
        self.host = resolved_host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = resolved_num_ctx
        self.keep_alive = keep_alive
        self.timeout = _resolve_timeout(timeout)
        self._client = (
            client if client is not None else httpx.Client(timeout=self.timeout)
        )

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise GenerationConfigError("Prompt cannot be empty")

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            response = self._client.post(f"{self.host}{CHAT_PATH}", json=body)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise GenerationAPIError(
                f"Ollama generation request to '{self.model}' at "
                f"'{self.host}' failed: {exc}"
            ) from exc

        message = payload.get("message") or {}
        text = strip_thinking(message.get("content") or "")

        if not text:
            raise GenerationAPIError(
                f"'{self.model}' returned no text for this prompt"
            )

        return text


def _resolve_timeout(timeout: float | None) -> float:
    raw = timeout if timeout is not None else OLLAMA_TIMEOUT

    if raw is None or raw == "":
        return DEFAULT_OLLAMA_TIMEOUT

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise GenerationConfigError(
            f"timeout must be a number, got {raw!r}"
        ) from exc

    if value <= 0:
        raise GenerationConfigError(f"timeout must be positive, got {value}")

    return value


__all__ = ["GenerationError", "OllamaGenerator", "strip_thinking"]