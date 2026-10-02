from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from knowledge_agent.config import EMBEDDING_FALLBACK_ORDER
from knowledge_agent.embeddings.health import ProviderHealth
from knowledge_agent.embeddings.ollama import OllamaEmbeddingProvider
from knowledge_agent.embeddings.provider import (
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingProvider,
    GeminiEmbeddingProvider,
)
from knowledge_agent.embeddings.space import SPACES, EmbeddingSpace

PROBE_TEXT = "probe"


@dataclass(frozen=True)
class Resolution:
    provider: EmbeddingProvider
    space: EmbeddingSpace
    note: str = ""
    fallback_used: bool = False


class LatchingProvider(EmbeddingProvider):
    """Latches on quota exhaustion so one exhausted provider costs one request.

    ``index_directory`` records a store failure once per batch without
    aborting, so without this every remaining batch would re-hit an exhausted
    daily quota.
    """

    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self._tripped = False
        self._error: EmbeddingError | None = None

    @property
    def space(self) -> EmbeddingSpace:
        return self._inner.space

    @property
    def tripped(self) -> bool:
        return self._tripped

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._guard(lambda: self._inner.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._guard(lambda: self._inner.embed_query(text))

    def _guard(self, call: Callable[[], object]) -> object:
        if self._tripped and self._error is not None:
            raise self._error

        try:
            result = call()
        except EmbeddingError as exc:
            self._tripped = True
            self._error = exc
            raise

        return result


def build_provider(space: EmbeddingSpace) -> EmbeddingProvider:
    if space.provider == "ollama":
        return OllamaEmbeddingProvider(model=space.model, space=space)

    return GeminiEmbeddingProvider(model=space.model)


def candidate_spaces(order: str | None = None) -> list[EmbeddingSpace]:
    keys = [key.strip() for key in (order or EMBEDDING_FALLBACK_ORDER or "").split(",") if key.strip()]

    if not keys:
        return list(SPACES.values())

    unknown = [key for key in keys if key not in SPACES]

    if unknown:
        raise EmbeddingConfigError(
            f"Unknown space(s) in EMBEDDING_FALLBACK_ORDER: {', '.join(unknown)}"
        )

    return [SPACES[key] for key in keys]


def resolve_provider(
    preferred: str | None = None,
    health: ProviderHealth | None = None,
    builder: Callable[[EmbeddingSpace], EmbeddingProvider] | None = None,
    now: datetime | None = None,
    probe: bool = True,
    order: str | None = None,
) -> Resolution:
    if preferred:
        return _pin(preferred, builder, health, now)

    store = health if health is not None else ProviderHealth()
    make = builder if builder is not None else build_provider

    skipped: list[str] = []
    failed: list[str] = []

    for space in candidate_spaces(order):
        record = store.active(space.key, now)

        if record is not None:
            skipped.append(
                f"{space.key} in cooldown for another "
                f"{record.seconds_remaining(now or datetime.now().astimezone())}s"
            )
            continue

        try:
            provider = make(space)
        except EmbeddingConfigError as exc:
            failed.append(f"{space.key}: {exc}")
            continue

        if not probe:
            return Resolution(provider=provider, space=space)

        try:
            provider.embed_query(PROBE_TEXT)
        except EmbeddingError as exc:
            failed.append(f"{space.key}: {exc}")
            store.record_from_message(space.key, str(exc), now=now)
            continue

        bypassed = failed + skipped
        note = f"using {space.key}"

        if bypassed:
            note = f"{note}; bypassed {', '.join(bypassed)}"

        return Resolution(
            provider=LatchingProvider(provider),
            space=space,
            note=note,
            fallback_used=bool(bypassed),
        )

    detail = "; ".join(failed) if failed else ""

    if detail:
        raise EmbeddingConfigError(
            f"No usable embedding space. Tried: {detail}"
            + (f". In cooldown: {', '.join(skipped)}" if skipped else "")
        )

    raise EmbeddingConfigError(
        "No usable embedding space. All candidates are in cooldown"
        + (f": {', '.join(skipped)}" if skipped else "")
    )


def resolve_collection(
    preferred: str | None = None,
    health: ProviderHealth | None = None,
    now: datetime | None = None,
    order: str | None = None,
) -> tuple[str, EmbeddingSpace]:
    """Pick a collection without touching the network.

    Skips spaces under an active cooldown but never probes, so ``count`` and
    ``spaces`` stay free of API calls.
    """
    if preferred:
        space = SPACES[preferred]
        return space.collection, space

    store = health if health is not None else ProviderHealth()
    candidates = candidate_spaces(order)

    for space in candidates:
        if store.active(space.key, now) is None:
            return space.collection, space

    return candidates[-1].collection, candidates[-1]


def _pin(
    preferred: str,
    builder: Callable[[EmbeddingSpace], EmbeddingProvider] | None,
    health: ProviderHealth | None,
    now: datetime | None,
) -> Resolution:
    if preferred not in SPACES:
        raise EmbeddingConfigError(
            f"Unknown space '{preferred}'. Known: {', '.join(SPACES)}"
        )

    space = SPACES[preferred]
    make = builder if builder is not None else build_provider
    provider = make(space)

    store = health if health is not None else ProviderHealth()
    record = store.active(space.key, now)
    note = (
        f"{space.key} is pinned but in cooldown "
        f"(retry in {record.seconds_remaining(now or datetime.now().astimezone())}s)"
        if record is not None
        else ""
    )

    return Resolution(
        provider=LatchingProvider(provider),
        space=space,
        note=note,
    )