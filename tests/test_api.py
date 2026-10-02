import pytest
from fastapi.testclient import TestClient

from knowledge_agent.embeddings.space import GEMINI_SPACE, NOMIC_SPACE, EmbeddingSpace
from knowledge_agent.rag.generator import (
    GenerationAPIError,
    GenerationConfigError,
    Generator,
)
from knowledge_agent.retrieval.vector_store import VectorStore, VectorStoreError
from knowledge_agent.schemas.retrieval import SearchResult

SPACES = {"gemini-3072": GEMINI_SPACE, "nomic-768": NOMIC_SPACE}


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
    def __init__(self, text="grounded answer [1]", error: Exception | None = None):
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
        "filename": "1706.03762v7.pdf",
        "page_number": 5,
        "chunk_index": 0,
    }
    defaults.update(kwargs)
    return SearchResult(**defaults)


class RecordingOpener:
    """Stands in for the cached store, recording what the API asked for."""

    def __init__(self, store, space=NOMIC_SPACE):
        self.store = store
        self.space = space
        self.keys: list[str] = []

    def __call__(self, space_key: str):
        self.keys.append(space_key)
        return self.store, self.space


def build_client(opener, generator):
    from knowledge_agent.api import create_app

    return TestClient(
        create_app(store_opener=opener, generator_factory=lambda: generator)
    )


@pytest.fixture
def store():
    return FakeStore([make_result()])


@pytest.fixture
def generator():
    return FakeGenerator()


@pytest.fixture
def opener(store):
    return RecordingOpener(store)


def test_health_reports_ok(opener, generator):
    response = build_client(opener, generator).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_an_answer(opener, generator):
    response = build_client(opener, generator).post("/ask", json={"question": "q"})

    assert response.status_code == 200
    assert response.json()["answer"] == "grounded answer [1]"


def test_ask_returns_the_question_asked(opener, generator):
    body = build_client(opener, generator).post("/ask", json={"question": "what is rag"}).json()

    assert body["question"] == "what is rag"


def test_ask_returns_filename_and_page_separately_from_the_answer(opener, generator):
    body = build_client(opener, generator).post("/ask", json={"question": "q"}).json()

    assert body["answer"] == "grounded answer [1]"
    assert body["citations"][0]["filename"] == "1706.03762v7.pdf"
    assert body["citations"][0]["page_number"] == 5
    assert "1706.03762v7.pdf" not in body["answer"]


def test_ask_reports_the_space(opener, generator):
    body = build_client(opener, generator).post("/ask", json={"question": "q"}).json()

    assert body["space"] == "nomic-768"


def test_ask_defaults_to_the_complete_index(opener, generator):
    build_client(opener, generator).post("/ask", json={"question": "q"})

    assert opener.keys == ["nomic-768"]


def test_ask_honours_an_explicit_space(store, generator):
    opener = RecordingOpener(store, space=GEMINI_SPACE)

    build_client(opener, generator).post("/ask", json={"question": "q", "space": "gemini-3072"})

    assert opener.keys == ["gemini-3072"]


def test_ask_passes_k_to_the_store(opener, generator):
    build_client(opener, generator).post("/ask", json={"question": "q", "k": 9})

    assert opener.store.searches[0][1] == 9


def test_the_question_reaches_the_store_verbatim(opener, generator):
    build_client(opener, generator).post("/ask", json={"question": "why is attention?"})

    assert opener.store.searches[0][0] == "why is attention?"


def test_the_context_is_prompted(opener, generator):
    opener.store.results = [make_result(content="the body")]

    build_client(opener, generator).post("/ask", json={"question": "q"})

    assert "the body" in generator.prompts[0]


def test_ask_rejects_a_blank_question(opener, generator):
    assert build_client(opener, generator).post("/ask", json={"question": "   "}).status_code == 422


def test_a_store_opening_failure_is_a_503(store, generator):
    def refuse(key: str):
        raise VectorStoreError("chroma down")

    response = build_client(refuse, generator).post("/ask", json={"question": "q"})

    assert response.status_code == 503


def test_a_generation_setup_failure_is_a_503(opener):
    from knowledge_agent.api import create_app

    def refuse():
        raise GenerationConfigError("No API key")

    app = create_app(store_opener=opener, generator_factory=refuse)

    assert TestClient(app).post("/ask", json={"question": "q"}).status_code == 503


def test_a_generation_failure_is_a_502(store):
    generator = FakeGenerator(error=GenerationAPIError("upstream 500"))
    client = build_client(RecordingOpener(store), generator)

    assert client.post("/ask", json={"question": "q"}).status_code == 502


def test_empty_retrieval_still_returns_200(generator):
    client = build_client(RecordingOpener(FakeStore([])), generator)

    response = client.post("/ask", json={"question": "q"})

    assert response.status_code == 200
    assert response.json()["citations"] == []


def test_the_min_distance_gate_is_reachable(generator):
    store = FakeStore([make_result(distance=0.99)])
    client = build_client(RecordingOpener(store), generator)

    body = client.post("/ask", json={"question": "q", "min_distance": 0.5}).json()

    assert body["citations"] == []
    assert generator.prompts == []
