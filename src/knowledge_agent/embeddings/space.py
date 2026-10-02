from dataclasses import dataclass

GEMINI_DIMENSIONS = 3072
NOMIC_DIMENSIONS = 768

QUERY_INSTRUCTION = "search_query: "
DOCUMENT_INSTRUCTION = "search_document: "


@dataclass(frozen=True)
class EmbeddingSpace:
    """A mutually incompatible embedding space.

    Two providers may emit vectors of the same width and still be unusable
    together: separately trained models share no coordinate basis, so cosine
    distance across them is meaningless. A space therefore owns a collection,
    and identity is tracked by ``key`` rather than by ``dimensions``.

    ``query_instruction`` and ``document_instruction`` are prefixes that some
    models require. Nomic-embed-text is trained with them, so omitting one
    measurably degrades retrieval rather than merely being untidy.
    """

    key: str
    provider: str
    model: str
    dimensions: int
    collection: str
    query_instruction: str = ""
    document_instruction: str = ""


GEMINI_SPACE = EmbeddingSpace(
    key="gemini-3072",
    provider="gemini",
    model="gemini-embedding-001",
    dimensions=GEMINI_DIMENSIONS,
    collection="knowledge",
)

NOMIC_SPACE = EmbeddingSpace(
    key="nomic-768",
    provider="ollama",
    model="nomic-embed-text-v2-moe",
    dimensions=NOMIC_DIMENSIONS,
    collection="knowledge--nomic-768",
    query_instruction=QUERY_INSTRUCTION,
    document_instruction=DOCUMENT_INSTRUCTION,
)

SPACES: dict[str, EmbeddingSpace] = {
    GEMINI_SPACE.key: GEMINI_SPACE,
    NOMIC_SPACE.key: NOMIC_SPACE,
}

SPACE_KEYS: tuple[str, ...] = tuple(SPACES)

LEGACY_COLLECTIONS: dict[str, EmbeddingSpace] = {
    GEMINI_SPACE.collection: GEMINI_SPACE,
}


def space_key(model: str, dimensions: int) -> str:
    return f"{model}-{dimensions}"


def space_for(model: str, dimensions: int, provider: str) -> EmbeddingSpace:
    """Resolve the registered space for a provider, or synthesise one.

    Matching is on provider, model, and width rather than on a key built from
    the model name, because registry keys are hand-chosen short names
    (``gemini-3072``) and are not derivable from the model string.
    """
    for space in SPACES.values():
        if (space.provider, space.model, space.dimensions) == (provider, model, dimensions):
            return space

    return EmbeddingSpace(
        key=space_key(model, dimensions),
        provider=provider,
        model=model,
        dimensions=dimensions,
        collection=f"knowledge--{provider}-{space_key(model, dimensions)}",
        query_instruction=QUERY_INSTRUCTION if provider == "ollama" else "",
        document_instruction=DOCUMENT_INSTRUCTION if provider == "ollama" else "",
    )
