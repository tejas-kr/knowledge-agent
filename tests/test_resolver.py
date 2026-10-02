import pytest

from knowledge_agent.embeddings.health import ProviderHealth
from knowledge_agent.embeddings.provider import (
    EmbeddingConfigError,
    EmbeddingProvider,
    EmbeddingQuotaError,
)
from knowledge_agent.embeddings.resolver import (
    LatchingProvider,
    build_provider,
    candidate_spaces,
    resolve_collection,
    resolve_provider,
)
from knowledge_agent.embeddings.space import GEMINI_SPACE, NOMIC_SPACE, SPACES


class FakeProvider(EmbeddingProvider):
    def __init__(self, space, error: Exception | None = None, dims: int = 2):
        self._space = space
        self.error = error
        self.dims = dims
        self.document_calls = 0
        self.query_calls: list[str] = []

    @property
    def space(self):
        return self._space

    def embed_documents(self, texts):
        if self.error:
            raise self.error
        self.document_calls += 1
        return [[0.0] * self.dims for _ in texts]

    def embed_query(self, text):
        if self.error:
            raise self.error
        self.query_calls.append(text)
        return [0.0] * self.dims


def health(tmp_path) -> ProviderHealth:
    return ProviderHealth(tmp_path / "provider_health.json")


def test_gemini_is_the_default_first_choice(tmp_path):
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: FakeProvider(space),
    )

    assert resolution.space.key == "gemini-3072"


def test_quota_exhaustion_falls_back_to_nomic(tmp_path):
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: FakeProvider(
            space, EmbeddingQuotaError("daily cap") if space.provider == "gemini" else None
        ),
    )

    assert resolution.space.key == "nomic-768"


def test_fallback_is_reported_as_a_note(tmp_path):
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: FakeProvider(
            space, EmbeddingQuotaError("daily cap") if space.provider == "gemini" else None
        ),
    )

    assert "gemini-3072" in resolution.note


def test_exhausted_space_is_recorded_for_next_time(tmp_path):
    store = health(tmp_path)
    resolve_provider(
        health=store,
        builder=lambda space: FakeProvider(
            space, EmbeddingQuotaError("daily cap") if space.provider == "gemini" else None
        ),
    )

    assert store.active("gemini-3072") is not None


def test_a_cooldown_skips_the_probe_entirely(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "daily cap", hours=24)
    built: list[str] = []

    resolution = resolve_provider(
        health=store,
        builder=lambda space: built.append(space.key) or FakeProvider(space),
    )

    assert resolution.space.key == "nomic-768"
    assert built == ["nomic-768"]


def test_a_cooldown_makes_gemini_free_next_run(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "daily cap", hours=24)
    gemini = FakeProvider(GEMINI_SPACE)

    resolve_provider(
        health=store,
        builder=lambda space: gemini if space.provider == "gemini" else FakeProvider(space),
    )

    assert gemini.query_calls == []


def test_the_primary_needs_no_probe_on_a_healthy_run(tmp_path):
    provider = FakeProvider(GEMINI_SPACE)
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: provider if space.provider == "gemini" else FakeProvider(space),
    )

    assert resolution.fallback_used is False
    assert resolution.note == "using gemini-3072"


def test_fallback_is_flagged(tmp_path):
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: FakeProvider(
            space, EmbeddingQuotaError("daily cap") if space.provider == "gemini" else None
        ),
    )

    assert resolution.fallback_used is True


def test_a_pin_never_reports_a_fallback(tmp_path):
    resolution = resolve_provider(
        "gemini-3072", health=health(tmp_path), builder=lambda s: FakeProvider(s)
    )

    assert resolution.fallback_used is False


def test_a_pinned_provider_is_still_latched(tmp_path):
    """A pin chooses the space; it does not licence retrying a dead provider."""
    provider = FakeProvider(GEMINI_SPACE, EmbeddingQuotaError("cap"))
    resolution = resolve_provider(
        "gemini-3072", health=health(tmp_path), builder=lambda s: provider
    )

    for _ in range(3):
        with pytest.raises(EmbeddingQuotaError):
            resolution.provider.embed_documents(["a"])


def test_a_config_error_also_falls_back(tmp_path):
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: (
            (_ for _ in ()).throw(EmbeddingConfigError("no api key"))
            if space.provider == "gemini"
            else FakeProvider(space)
        ),
    )

    assert resolution.space.key == "nomic-768"


def test_a_pinned_space_skips_the_cooldown(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "daily cap", hours=24)
    provider = FakeProvider(GEMINI_SPACE)

    resolution = resolve_provider("gemini-3072", health=store, builder=lambda s: provider)

    assert resolution.space.key == "gemini-3072"
    assert provider.query_calls == []


def test_a_pinned_space_reports_its_cooldown(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "daily cap", hours=24)

    resolution = resolve_provider("gemini-3072", health=store, builder=lambda s: FakeProvider(s))

    assert "cooldown" in resolution.note


def test_a_pinned_space_does_not_record_health(tmp_path):
    store = health(tmp_path)
    resolve_provider(
        "gemini-3072",
        health=store,
        builder=lambda s: FakeProvider(s, EmbeddingQuotaError("cap")),
    )

    assert store.active("gemini-3072") is None


def test_an_unknown_pinned_space_is_rejected(tmp_path):
    with pytest.raises(EmbeddingConfigError, match="Unknown space"):
        resolve_provider("nope-1024", health=health(tmp_path), builder=lambda s: None)


def test_every_candidate_failing_names_them(tmp_path):
    with pytest.raises(EmbeddingConfigError) as excinfo:
        resolve_provider(
            health=health(tmp_path),
            builder=lambda space: FakeProvider(space, EmbeddingQuotaError("down")),
        )

    assert "gemini-3072" in str(excinfo.value)
    assert "nomic-768" in str(excinfo.value)


def test_all_candidates_in_cooldown_says_so(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "cap", hours=24)
    store.record_exhausted("nomic-768", "cap", hours=24)

    with pytest.raises(EmbeddingConfigError, match="cooldown"):
        resolve_provider(health=store, builder=lambda s: FakeProvider(s))


def test_fallback_order_is_honoured(tmp_path):
    resolution = resolve_provider(
        health=health(tmp_path),
        builder=lambda space: FakeProvider(space),
        order="nomic-768,gemini-3072",
    )

    assert resolution.space.key == "nomic-768"


def test_default_order_is_the_registry_order():
    assert candidate_spaces() == [GEMINI_SPACE, NOMIC_SPACE]


def test_an_unknown_fallback_order_entry_is_rejected():
    with pytest.raises(EmbeddingConfigError, match="Unknown space"):
        candidate_spaces("gemini-3072,bogus")


def test_fallback_order_ignores_blanks():
    assert candidate_spaces(" gemini-3072 , , nomic-768 ") == [GEMINI_SPACE, NOMIC_SPACE]


def test_ollama_space_builds_an_ollama_provider():
    assert type(build_provider(NOMIC_SPACE)).__name__ == "OllamaEmbeddingProvider"


def test_collection_follows_the_resolved_space(tmp_path):
    collection, space = resolve_collection(health=health(tmp_path))

    assert (collection, space.key) == ("knowledge", "gemini-3072")


def test_collection_resolution_skips_a_cooldown(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "cap", hours=24)

    collection, space = resolve_collection(health=store)

    assert (collection, space.key) == ("knowledge--nomic-768", "nomic-768")


def test_collection_resolution_never_probes(tmp_path):
    resolution = resolve_collection(health=health(tmp_path))

    assert resolution[1].key == GEMINI_SPACE.key


def test_pinned_collection_ignores_cooldown(tmp_path):
    store = health(tmp_path)
    store.record_exhausted("gemini-3072", "cap", hours=24)

    collection, space = resolve_collection("gemini-3072", health=store)

    assert (collection, space.key) == ("knowledge", "gemini-3072")


def test_a_latch_lets_only_one_call_through():
    provider = FakeProvider(GEMINI_SPACE, EmbeddingQuotaError("cap"))
    latch = LatchingProvider(provider)

    with pytest.raises(EmbeddingQuotaError):
        latch.embed_documents(["a"])

    assert provider.document_calls == 0
    assert latch.tripped is True


def test_a_latched_provider_repeats_without_calling_again():
    provider = FakeProvider(GEMINI_SPACE, EmbeddingQuotaError("cap"))
    latch = LatchingProvider(provider)

    for _ in range(3):
        with pytest.raises(EmbeddingQuotaError):
            latch.embed_documents(["a"])


def test_a_latch_passes_success_through():
    latch = LatchingProvider(FakeProvider(GEMINI_SPACE))

    assert latch.embed_documents(["a"]) == [[0.0, 0.0]]
    assert latch.tripped is False


def test_a_latch_reports_its_inner_space():
    assert LatchingProvider(FakeProvider(NOMIC_SPACE)).space.key == "nomic-768"


def test_resolved_providers_are_latched(tmp_path):
    resolution = resolve_provider(health=health(tmp_path), builder=lambda s: FakeProvider(s))

    assert isinstance(resolution.provider, LatchingProvider)


def test_a_pin_is_latched(tmp_path):
    resolution = resolve_provider(
        "gemini-3072", health=health(tmp_path), builder=lambda s: FakeProvider(s)
    )

    assert isinstance(resolution.provider, LatchingProvider)


def test_the_registry_holds_exactly_two_spaces():
    assert set(SPACES) == {"gemini-3072", "nomic-768"}


def test_the_two_collections_are_distinct():
    assert SPACES["gemini-3072"].collection != SPACES["nomic-768"].collection
