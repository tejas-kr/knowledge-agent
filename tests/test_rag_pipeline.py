import pytest

from knowledge_agent.embeddings.space import NOMIC_SPACE
from knowledge_agent.rag.generator import GenerationAPIError, Generator
from knowledge_agent.rag.pipeline import (
    NO_CONTEXT_ANSWER,
    RAGError,
    answer_question,
    build_citations,
)
from knowledge_agent.rag.prompt import CITATION_INSTRUCTION
from knowledge_agent.retrieval.vector_store import VectorStore, VectorStoreError
from knowledge_agent.schemas.retrieval import SearchResult


class FakeStore(VectorStore):
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.searches: list[tuple] = []

    def search(self, query, k=5, where=None):
        self.searches.append((query, k, where))
        if self.error:
            raise self.error
        return list(self.results)

    def add_chunks(self, chunks, embeddings=None):
        return []

    def count(self):
        return len(self.results)

    def delete_document(self, document_id):
        return None

    def existing_ids(self):
        return set()


class FakeGenerator(Generator):
    def __init__(self, text="an answer [1]", error: Exception | None = None):
        self.text = text
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.text


def make_result(**kwargs) -> SearchResult:
    defaults = {
        "chunk_id": "doc:1:0",
        "content": "body text",
        "distance": 0.2,
        "document_id": "doc",
        "filename": "doc.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }
    defaults.update(kwargs)
    return SearchResult(**defaults)


def test_the_question_reaches_the_store_verbatim():
    store = FakeStore([make_result()])
    answer_question("why is the sky blue?", store, FakeGenerator(), k=3)

    assert store.searches == [("why is the sky blue?", 3, None)]


def test_the_generated_prompt_carries_context_and_question():
    generator = FakeGenerator()
    store = FakeStore([make_result(content="the context body")])

    answer_question("the question", store, generator, space=NOMIC_SPACE)

    prompt = generator.prompts[0]
    assert "the context body" in prompt
    assert "the question" in prompt
    assert "{context}" not in prompt
    assert "{question}" not in prompt


def test_the_prompt_asks_the_model_to_cite():
    generator = FakeGenerator()
    store = FakeStore([make_result()])

    answer_question("q", store, generator)

    assert CITATION_INSTRUCTION in generator.prompts[0]


def test_the_citation_ask_can_be_switched_off():
    generator = FakeGenerator()
    store = FakeStore([make_result()])

    answer_question("q", store, generator, cite_instruction=False)

    assert CITATION_INSTRUCTION not in generator.prompts[0]


def test_the_prompt_carries_the_filename_and_page():
    generator = FakeGenerator()
    store = FakeStore([make_result(filename="1706.03762v7.pdf", page_number=5)])

    answer_question("q", store, generator)

    assert "1706.03762v7.pdf" in generator.prompts[0]
    assert "page 5" in generator.prompts[0]


def test_citations_carry_the_filename_and_page():
    store = FakeStore(
        [make_result(filename="1706.03762v7.pdf", page_number=5, distance=0.11)]
    )

    answer = answer_question("q", store, FakeGenerator())

    citation = answer.citations[0]
    assert citation.filename == "1706.03762v7.pdf"
    assert citation.page_number == 5


def test_citation_indices_start_at_one():
    store = FakeStore([make_result(), make_result(chunk_index=1)])

    answer = answer_question("q", store, FakeGenerator())

    assert [c.index for c in answer.citations] == [1, 2]


def test_a_cited_marker_marks_its_citation():
    store = FakeStore([make_result(), make_result(chunk_index=1)])
    generator = FakeGenerator(text="only the first source matters [1]")

    answer = answer_question("q", store, generator)

    assert [c.cited for c in answer.citations] == [True, False]


def test_an_uncited_chunk_is_still_reported():
    """Provenance must survive a model that ignores the markers."""
    store = FakeStore([make_result(), make_result(chunk_index=1)])
    generator = FakeGenerator(text="no markers at all")

    answer = answer_question("q", store, generator)

    assert len(answer.citations) == 2
    assert [c.cited for c in answer.citations] == [False, False]


def test_the_space_is_reported_back():
    answer = answer_question("q", FakeStore([make_result()]), FakeGenerator(), space=NOMIC_SPACE)

    assert answer.space == "nomic-768"


def test_the_top_distance_is_reported():
    store = FakeStore([make_result(distance=0.42), make_result(distance=0.9)])

    answer = answer_question("q", store, FakeGenerator())

    assert answer.top_distance == 0.42


def test_empty_retrieval_answers_without_calling_the_model():
    generator = FakeGenerator()

    answer = answer_question("q", FakeStore([]), generator)

    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.citations == []
    assert generator.prompts == []


def test_empty_retrieval_reports_no_top_distance():
    assert answer_question("q", FakeStore([]), FakeGenerator()).top_distance is None


def test_a_min_distance_gate_skips_generation():
    store = FakeStore([make_result(distance=0.95)])
    generator = FakeGenerator()

    answer = answer_question("q", store, generator, min_distance=0.5)

    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.citations == []
    assert generator.prompts == []


def test_a_min_distance_gate_passes_a_close_match():
    store = FakeStore([make_result(distance=0.10)])
    generator = FakeGenerator()

    answer = answer_question("q", store, generator, min_distance=0.5)

    assert answer.answer == "an answer [1]"


def test_context_chunks_counts_what_was_kept():
    store = FakeStore([make_result(), make_result(chunk_index=1)])

    answer = answer_question("q", store, FakeGenerator())

    assert answer.context_chunks == 2


def test_an_oversized_context_is_trimmed():
    store = FakeStore([make_result(content="x" * 4000) for _ in range(5)])
    generator = FakeGenerator()

    answer = answer_question("q", store, generator, max_context_chars=500)

    assert answer.context_chunks < 5


def test_trimming_keeps_the_best_ranked_chunk():
    best = make_result(chunk_id="best", content="x" * 4000)
    store = FakeStore([best, make_result(content="y" * 4000)])
    generator = FakeGenerator()

    answer = answer_question("q", store, generator, max_context_chars=500)

    assert answer.citations[0].chunk_id == "best"


def test_an_empty_question_is_rejected():
    with pytest.raises(RAGError, match="Question"):
        answer_question("  ", FakeStore(), FakeGenerator())


def test_a_non_positive_k_is_rejected():
    with pytest.raises(RAGError, match="k must be positive"):
        answer_question("q", FakeStore(), FakeGenerator(), k=0)


def test_a_non_positive_budget_is_rejected():
    with pytest.raises(RAGError, match="max_context_chars"):
        answer_question("q", FakeStore(), FakeGenerator(), max_context_chars=0)


def test_a_retrieval_failure_is_wrapped():
    store = FakeStore(error=VectorStoreError("chroma down"))

    with pytest.raises(RAGError, match="Retrieval failed"):
        answer_question("q", store, FakeGenerator())


def test_a_retrieval_failure_preserves_its_cause():
    cause = VectorStoreError("chroma down")

    with pytest.raises(RAGError) as excinfo:
        answer_question("q", FakeStore(error=cause), FakeGenerator())

    assert excinfo.value.__cause__ is cause


def test_a_generation_failure_is_wrapped():
    generator = FakeGenerator(error=GenerationAPIError("upstream 500"))

    with pytest.raises(RAGError, match="Answer generation failed"):
        answer_question("q", FakeStore([make_result()]), generator)


def test_the_answer_and_the_citations_are_separate_fields():
    store = FakeStore([make_result(filename="a.pdf", page_number=2)])
    generator = FakeGenerator(text="the model wrote this [1]")

    answer = answer_question("q", store, generator)

    assert answer.answer == "the model wrote this [1]"
    assert answer.citations[0].filename == "a.pdf"
    assert "a.pdf" not in answer.answer


def test_citations_are_built_from_results_not_from_the_answer():
    """A model inventing [9] must not create a citation."""
    citations = build_citations([make_result()], "invented marker [9] here")

    assert len(citations) == 1
    assert citations[0].index == 1
    assert citations[0].cited is False
