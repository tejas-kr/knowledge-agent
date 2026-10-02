from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Provenance for one retrieved chunk.

    ``filename`` and ``page_number`` are read from the store, never parsed out
    of generated text, so they cannot be hallucinated.
    """

    index: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    distance: float
    cited: bool = False


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    space: str = ""
    top_distance: float | None = None
    context_chunks: int = 0


class AskRequest(BaseModel):
    question: str
    k: int = 5
    space: str | None = None
    min_distance: float | None = None
