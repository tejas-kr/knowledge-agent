from knowledge_agent.embeddings.provider import GeminiEmbeddingProvider
from knowledge_agent.embeddings.space import GEMINI_SPACE, NOMIC_SPACE, space_for


def test_gemini_model_resolves_to_the_registered_space():
    assert space_for("gemini-embedding-001", 3072, "gemini") is GEMINI_SPACE


def test_nomic_model_resolves_to_the_registered_space():
    assert space_for("nomic-embed-text-v2-moe", 768, "ollama") is NOMIC_SPACE


def test_nomic_reports_its_own_width():
    assert NOMIC_SPACE.dimensions == 768


def test_nomic_requires_both_task_prefixes():
    assert NOMIC_SPACE.query_instruction == "search_query: "
    assert NOMIC_SPACE.document_instruction == "search_document: "


def test_an_unknown_model_gets_its_own_collection():
    space = space_for("mystery-model", 1024, "ollama")

    assert space.key == "mystery-model-1024"
    assert space.collection not in {GEMINI_SPACE.collection, NOMIC_SPACE.collection}


def test_an_unknown_width_gets_its_own_collection():
    space = space_for("gemini-embedding-001", 768, "gemini")

    assert space.collection != GEMINI_SPACE.collection


def test_a_synthesised_ollama_space_keeps_the_query_instruction():
    assert space_for("mystery-model", 1024, "ollama").query_instruction


def test_a_synthesised_gemini_space_has_no_query_instruction():
    assert space_for("mystery-model", 1024, "gemini").query_instruction == ""


def test_gemini_provider_reports_the_registered_space():
    assert GeminiEmbeddingProvider(
        model="gemini-embedding-001", api_key="k"
    ).space is GEMINI_SPACE
