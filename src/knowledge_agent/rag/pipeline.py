from collections.abc import Sequence

from knowledge_agent.config import (
    RAG_CITE_INSTRUCTION,
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOP_K,
)
from knowledge_agent.embeddings.space import EmbeddingSpace
from knowledge_agent.rag.generator import GenerationError, Generator
from knowledge_agent.rag.prompt import (
    cited_indexes,
    format_context,
    render_prompt,
    trim_context,
)
from knowledge_agent.retrieval.vector_store import VectorStore, VectorStoreError
from knowledge_agent.schemas.rag import Citation, RAGAnswer
from knowledge_agent.schemas.retrieval import SearchResult

NO_CONTEXT_ANSWER = (
    "The indexed documents do not contain enough information to answer that question."
)


class RAGError(Exception):
    pass


def answer_question(
    question: str,
    store: VectorStore,
    generator: Generator,
    space: EmbeddingSpace | None = None,
    k: int = RAG_TOP_K,
    max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
    min_distance: float | None = None,
    cite_instruction: bool = RAG_CITE_INSTRUCTION,
) -> RAGAnswer:
    """Retrieve, prompt, and generate a grounded answer with citations."""
    if not question.strip():
        raise RAGError("Question cannot be empty")

    if k <= 0:
        raise RAGError(f"k must be positive, got {k}")

    if max_context_chars <= 0:
        raise RAGError(f"max_context_chars must be positive, got {max_context_chars}")

    results = _retrieve(question, store, k)
    space_key = space.key if space is not None else ""

    if not results:
        return RAGAnswer(
            question=question,
            answer=NO_CONTEXT_ANSWER,
            citations=[],
            space=space_key,
            top_distance=None,
            context_chunks=0,
        )

    top_distance = results[0].distance

    if min_distance is not None and top_distance > min_distance:
        return RAGAnswer(
            question=question,
            answer=NO_CONTEXT_ANSWER,
            citations=[],
            space=space_key,
            top_distance=top_distance,
            context_chunks=0,
        )

    kept: Sequence[SearchResult] = trim_context(results, max_context_chars)
    prompt = render_prompt(
        format_context(kept), question, cite_instruction=cite_instruction
    )
    answer = _generate(prompt, generator)

    return RAGAnswer(
        question=question,
        answer=answer,
        citations=build_citations(kept, answer),
        space=space_key,
        top_distance=top_distance,
        context_chunks=len(kept),
    )


def build_citations(
    results: Sequence[SearchResult], answer: str
) -> list[Citation]:
    """Map retrieved chunks to citations, flagging the ones the answer used.

    Every retrieved chunk is returned whether or not the model cited it, so a
    model that ignores the markers loses no provenance.
    """
    used = cited_indexes(answer)

    return [
        Citation(
            index=index,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            distance=result.distance,
            cited=index in used,
        )
        for index, result in enumerate(results, start=1)
    ]


def _retrieve(question: str, store: VectorStore, k: int) -> list[SearchResult]:
    try:
        return list(store.search(question, k=k))
    except VectorStoreError as exc:
        raise RAGError(f"Retrieval failed: {exc}") from exc


def _generate(prompt: str, generator: Generator) -> str:
    try:
        return generator.generate(prompt)
    except GenerationError as exc:
        raise RAGError(f"Answer generation failed: {exc}") from exc
