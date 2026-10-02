import pytest

from knowledge_agent.rag.generator import (
    GenerationAPIError,
    GenerationConfigError,
    GenerationError,
    Generator,
    GeminiGenerator,
)


class FakeModels:
    def __init__(self, text="an answer", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        if self.error:
            raise self.error

        return type("Response", (), {"text": self.text})()


class FakeClient:
    def __init__(self, text="an answer", error: Exception | None = None):
        self.models = FakeModels(text, error)


def build(client=None, **kwargs) -> GeminiGenerator:
    return GeminiGenerator(
        model="gemini-3.8-flash", client=client or FakeClient(), **kwargs
    )


def test_the_generator_is_an_abstract_base():
    assert issubclass(GeminiGenerator, Generator)

    with pytest.raises(TypeError):
        Generator()


def test_the_prompt_is_sent_verbatim():
    client = FakeClient()
    build(client).generate("PROMPT AS GIVEN")

    assert client.models.calls[0]["contents"] == "PROMPT AS GIVEN"


def test_the_model_is_passed_through():
    client = FakeClient()
    build(client).generate("q")

    assert client.models.calls[0]["model"] == "gemini-3.8-flash"


def test_generation_is_deterministic_by_default():
    client = FakeClient()
    build(client).generate("q")

    assert client.models.calls[0]["config"].temperature == 0.0


def test_temperature_is_configurable():
    client = FakeClient()
    build(client, temperature=0.7).generate("q")

    assert client.models.calls[0]["config"].temperature == 0.7


def test_negative_temperature_is_rejected():
    with pytest.raises(GenerationConfigError, match="temperature"):
        build(temperature=-1)


def test_the_response_text_is_returned():
    assert build(FakeClient(text="grounded answer")).generate("q") == "grounded answer"


def test_surrounding_whitespace_is_stripped():
    assert build(FakeClient(text="  padded  ")).generate("q") == "padded"


def test_an_empty_prompt_is_rejected():
    with pytest.raises(GenerationConfigError, match="Prompt"):
        build().generate("   ")


def test_an_api_failure_is_wrapped():
    client = FakeClient(error=RuntimeError("upstream 500"))

    with pytest.raises(GenerationAPIError) as excinfo:
        build(client).generate("q")

    assert "gemini-3.8-flash" in str(excinfo.value)


def test_an_api_failure_preserves_its_cause():
    cause = RuntimeError("upstream 500")

    with pytest.raises(GenerationAPIError) as excinfo:
        build(FakeClient(error=cause)).generate("q")

    assert excinfo.value.__cause__ is cause


def test_an_empty_response_is_an_error():
    with pytest.raises(GenerationAPIError, match="no text"):
        build(FakeClient(text="")).generate("q")


def test_a_missing_api_key_is_reported(monkeypatch):
    monkeypatch.setattr("knowledge_agent.rag.generator.GEMINI_API_KEY", None)

    with pytest.raises(GenerationConfigError, match="No API key"):
        GeminiGenerator()


def test_the_error_hierarchy_is_rooted_in_one_base():
    assert issubclass(GenerationConfigError, GenerationError)
    assert issubclass(GenerationAPIError, GenerationError)
    assert not issubclass(GenerationConfigError, GenerationAPIError)
