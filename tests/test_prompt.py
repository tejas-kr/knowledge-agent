import pytest

from knowledge_agent.rag.prompt import (
    CITATION_INSTRUCTION,
    SYSTEM_PROMPT,
    cited_indexes,
    format_context,
    render_prompt,
    trim_context,
)
from knowledge_agent.schemas.retrieval import SearchResult


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


def test_the_template_carries_both_placeholders():
    assert "{context}" in SYSTEM_PROMPT
    assert "{question}" in SYSTEM_PROMPT


def test_both_placeholders_are_filled():
    rendered = render_prompt("CTX", "QST")

    assert "CTX" in rendered
    assert "QST" in rendered
    assert "{context}" not in rendered
    assert "{question}" not in rendered


def test_latex_braces_in_context_do_not_crash():
    """This corpus is arXiv PDFs; str.format would raise on every chunk."""
    latex = r"$\frac{a}{b}$ and \textbf{x} plus a bare { brace"

    rendered = render_prompt(latex, "why?")

    assert latex in rendered


def test_a_chunk_containing_the_question_placeholder_is_not_resubstituted():
    rendered = render_prompt("literal {question} stays", "real question")

    assert "literal {question} stays" in rendered
    assert "real question" in rendered


def test_inserted_content_is_not_rescanned_for_placeholders():
    rendered = render_prompt("{context} and {question}", "q")

    assert "{context} and {question}" in rendered


def test_context_is_numbered_from_one():
    block = format_context([make_result(), make_result()])

    assert block.startswith("[1] (doc.pdf, page 1)")


def test_context_numbers_follow_rank_order():
    block = format_context(
        [make_result(filename="a.pdf", page_number=3), make_result(filename="b.pdf")]
    )

    assert "[1] (a.pdf, page 3)" in block
    assert "[2] (b.pdf, page 1)" in block


def test_context_includes_every_retrieved_chunk():
    block = format_context([make_result(content="first"), make_result(content="second")])

    assert "first" in block
    assert "second" in block


def test_context_is_empty_for_no_results():
    assert format_context([]) == ""


def test_trimming_drops_the_lowest_ranked_chunk_first():
    results = [make_result(content="x" * 200) for _ in range(3)]

    kept = trim_context(results, 100)

    assert len(kept) < 3
    assert kept[0] is results[0]


def test_trimming_keeps_everything_that_already_fits():
    results = [make_result(content="short")]

    assert trim_context(results, 100_000) == results


def test_trimming_never_empties_the_context():
    """One oversized chunk is still better than answering with no context."""
    kept = trim_context([make_result(content="x" * 5000)], 100)

    assert len(kept) == 1


def test_cited_indexes_reads_a_single_marker():
    assert cited_indexes("RAG combines retrieval [1] with generation.") == {1}


def test_cited_indexes_reads_several_markers():
    assert cited_indexes("first [1] and second [3]") == {1, 3}


def test_cited_indexes_reads_a_comma_group():
    assert cited_indexes("both agree [1, 2]") == {1, 2}


def test_cited_indexes_tolerates_spaces_in_a_group():
    assert cited_indexes("both agree [1, 2]") == {1, 2}


def test_cited_indexes_ignores_prose_numbers():
    assert cited_indexes("there are 12 layers") == set()


def test_cited_indexes_ignores_an_empty_bracket():
    assert cited_indexes("nothing cited []") == set()


def test_cited_indexes_ignores_non_numeric_bracket_text():
    assert cited_indexes("see [Appendix] for details") == set()


def test_the_citation_instruction_asks_for_bracketed_markers():
    """Small local models do not infer [n] markers on their own."""
    assert "[1]" in CITATION_INSTRUCTION


def test_the_citation_instruction_is_added_by_default():
    rendered = render_prompt("CTX", "QST")

    assert CITATION_INSTRUCTION in rendered
    assert "QST" in rendered


def test_the_citation_instruction_can_be_turned_off():
    rendered = render_prompt("CTX", "QST", cite_instruction=False)

    assert CITATION_INSTRUCTION not in rendered
    assert "QST" in rendered


def test_the_system_prompt_is_not_rewritten_by_the_citation_instruction():
    """It must stay byte-identical to the version the user specified."""
    assert CITATION_INSTRUCTION not in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.endswith("{question}\n")
