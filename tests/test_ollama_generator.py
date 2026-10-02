import httpx
import pytest

from knowledge_agent.rag.generator import (
    GenerationAPIError,
    GenerationConfigError,
    Generator,
    build_generator,
)
from knowledge_agent.rag.ollama import OllamaGenerator, strip_thinking


class FakeResponse:
    def __init__(self, payload: dict | None = None, status: int = 200):
        self._payload = (
            payload if payload is not None else {"message": {"content": ""}}
        )
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, content="grounded answer [1]", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def post(self, url, json=None, **kwargs):
        self.calls.append({"url": url, "json": json})

        if self.error:
            raise self.error

        return FakeResponse({"message": {"role": "assistant", "content": self.content}})

    @property
    def requests(self):
        return [(call["url"], call["json"]) for call in self.calls]


def build(client=None, **kwargs) -> OllamaGenerator:
    return OllamaGenerator(
        model="qwen3:1.7b",
        host="http://localhost:11434",
        client=client if client is not None else FakeHTTP(),
        **kwargs,
    )


def test_it_is_a_generator():
    assert issubclass(OllamaGenerator, Generator)


def test_it_posts_to_the_chat_endpoint():
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][0] == "http://localhost:11434/api/chat"


def test_it_uses_chat_not_the_raw_completion_endpoint():
    """An instruct model only applies its chat template on /api/chat."""
    http = FakeHTTP()

    build(http).generate("q")

    assert "/api/generate" not in http.requests[0][0]


def test_the_prompt_is_sent_as_one_user_message():
    http = FakeHTTP()

    build(http).generate("PROMPT AS GIVEN")

    assert http.requests[0][1]["messages"] == [
        {"role": "user", "content": "PROMPT AS GIVEN"}
    ]


def test_the_model_is_passed_through():
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][1]["model"] == "qwen3:1.7b"


def test_thinking_is_switched_off():
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][1]["think"] is False


def test_streaming_is_disabled():
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][1]["stream"] is False


def test_the_context_window_is_widened_past_the_default():
    """A full RAG prompt is ~3-4k tokens; Ollama's 4096 default can truncate."""
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][1]["options"]["num_ctx"] == 8192


def test_the_context_window_is_configurable():
    http = FakeHTTP()

    build(http, num_ctx=4096).generate("q")

    assert http.requests[0][1]["options"]["num_ctx"] == 4096


def test_generation_is_deterministic_by_default():
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][1]["options"]["temperature"] == 0.0


def test_keep_alive_is_set_so_the_model_stays_on_the_card():
    http = FakeHTTP()

    build(http).generate("q")

    assert http.requests[0][1]["keep_alive"] == "30m"


def test_the_answer_is_returned():
    assert build(FakeHTTP(content="grounded answer [1]")).generate("q") == "grounded answer [1]"


def test_citation_markers_survive_the_round_trip():
    assert "[1]" in build(FakeHTTP(content="claim [1] and [2]")).generate("q")


def test_thinking_blocks_are_stripped():
    content = "<think>reasoning about [3]</think>\nthe actual answer [1]"

    assert build(FakeHTTP(content=content)).generate("q") == "the actual answer [1]"


def test_an_unclosed_thinking_block_is_discarded_entirely():
    """A truncated reasoning block means there is no answer to trust."""
    assert strip_thinking("<think>drowned mid-reasoning") == ""


def test_an_unclosed_thinking_block_becomes_an_error():
    with pytest.raises(GenerationAPIError, match="no text"):
        build(FakeHTTP(content="<think>reasoning cut off here")).generate("q")


def test_thinking_is_stripped_with_a_generic_tag():
    assert strip_thinking("<thinking>x</thinking>answer") == "answer"


def test_answer_without_thinking_is_untouched():
    assert strip_thinking("plain answer") == "plain answer"


def test_an_empty_prompt_is_rejected():
    with pytest.raises(GenerationConfigError, match="Prompt"):
        build().generate("   ")


def test_a_connection_failure_is_wrapped():
    http = FakeHTTP(error=httpx.ConnectError("connection refused"))

    with pytest.raises(GenerationAPIError) as excinfo:
        build(http).generate("q")

    assert "qwen3:1.7b" in str(excinfo.value)


def test_a_connection_failure_preserves_its_cause():
    cause = httpx.ConnectError("connection refused")

    with pytest.raises(GenerationAPIError) as excinfo:
        build(FakeHTTP(error=cause)).generate("q")

    assert excinfo.value.__cause__ is cause


def test_an_http_error_status_is_wrapped():
    http = FakeHTTP()
    http.post = lambda url, json=None, **kw: FakeResponse(status=500)

    with pytest.raises(GenerationAPIError, match="failed"):
        build(http).generate("q")


def test_an_empty_answer_is_an_error():
    with pytest.raises(GenerationAPIError, match="no text"):
        build(FakeHTTP(content="")).generate("q")


def test_a_thinking_only_answer_is_an_error():
    with pytest.raises(GenerationAPIError, match="no text"):
        build(FakeHTTP(content="<think>only reasoning</think>")).generate("q")


def test_a_missing_message_key_is_an_error():
    http = FakeHTTP()
    http.post = lambda url, json=None, **kw: FakeResponse({})

    with pytest.raises(GenerationAPIError, match="no text"):
        build(http).generate("q")


def test_a_missing_host_is_reported():
    with pytest.raises(GenerationConfigError, match="No Ollama host"):
        OllamaGenerator(host="   ")


def test_an_empty_model_is_rejected():
    with pytest.raises(GenerationConfigError, match="model"):
        OllamaGenerator(model="  ", host="http://localhost:11434")


def test_a_non_positive_context_window_is_rejected():
    with pytest.raises(GenerationConfigError, match="num_ctx"):
        build(num_ctx=0)


def test_a_negative_temperature_is_rejected():
    with pytest.raises(GenerationConfigError, match="temperature"):
        build(temperature=-1)


def test_an_empty_keep_alive_is_rejected():
    with pytest.raises(GenerationConfigError, match="keep_alive"):
        build(keep_alive="")


def test_the_host_loses_its_trailing_slash():
    http = FakeHTTP()

    OllamaGenerator(
        model="qwen3:1.7b", host="http://localhost:11434/", client=http
    ).generate("q")

    assert http.requests[0][0] == "http://localhost:11434/api/chat"


def test_build_generator_defaults_to_ollama():
    from knowledge_agent.rag.ollama import OllamaGenerator as Concrete

    assert isinstance(build_generator(), Concrete)


def test_the_default_model_is_the_one_that_fits_a_4gb_card():
    """Regression guard: qwen3:8b would not fit, qwen3:1.7b does."""
    from knowledge_agent.config import OLLAMA_GENERATION_MODEL

    assert OLLAMA_GENERATION_MODEL == "qwen3:1.7b"


def test_the_backend_is_switchable_to_gemini():
    """Gemini stays reachable for quality comparison, just not the default."""
    assert build_generator("gemini") is not None


def test_build_generator_can_pick_gemini(monkeypatch):
    monkeypatch.setattr("knowledge_agent.rag.generator.GEMINI_API_KEY", "test-key")

    from knowledge_agent.rag.generator import GeminiGenerator

    assert isinstance(build_generator("gemini"), GeminiGenerator)


def test_build_generator_rejects_an_unknown_backend():
    with pytest.raises(GenerationConfigError, match="Unknown GENERATOR_BACKEND"):
        build_generator("llamafile")