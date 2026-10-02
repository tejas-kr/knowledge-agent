from abc import ABC, abstractmethod

from google import genai

from knowledge_agent.config import GENERATOR_BACKEND, GEMINI_API_KEY, GEMINI_MODEL

DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
GROUNDED_TEMPERATURE = 0.0


class GenerationError(Exception):
    pass


class GenerationConfigError(GenerationError):
    pass


class GenerationAPIError(GenerationError):
    pass


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiGenerator(Generator):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = GROUNDED_TEMPERATURE,
        client: genai.Client | None = None,
    ) -> None:
        self.model = model or GEMINI_MODEL or DEFAULT_GENERATION_MODEL

        if temperature < 0:
            raise GenerationConfigError(
                f"temperature cannot be negative, got {temperature}"
            )

        self.temperature = temperature

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or GEMINI_API_KEY

        if not resolved_key:
            raise GenerationConfigError(
                "No API key: set GEMINI_API_KEY in the environment or pass api_key=..."
            )

        self._client = genai.Client(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise GenerationConfigError("Prompt cannot be empty")

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature,
                ),
            )
        except Exception as exc:
            raise GenerationAPIError(
                f"Generation request to '{self.model}' failed: {exc}"
            ) from exc

        text = (getattr(response, "text", None) or "").strip()

        if not text:
            raise GenerationAPIError(
                f"'{self.model}' returned no text for this prompt"
            )

        return text


def build_generator(backend: str | None = None) -> Generator:
    """Pick a generator from ``GENERATOR_BACKEND``.

    The local backend is the default: the Gemini free tier answers with
    transient ``503 UNAVAILABLE`` under load and needs no quota, so a 4 GB card
    with a 1.7B model is the more reliable path even though it is slower.
    """
    from knowledge_agent.rag.ollama import OllamaGenerator

    resolved = (backend or GENERATOR_BACKEND or "ollama").strip().lower()

    if resolved == "ollama":
        return OllamaGenerator()
    if resolved == "gemini":
        return GeminiGenerator()

    raise GenerationConfigError(
        f"Unknown GENERATOR_BACKEND '{resolved}': expected 'ollama' or 'gemini'"
    )
