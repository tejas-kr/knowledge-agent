import random
import string

import pymupdf
import pytest

from knowledge_agent.ingestion.chunker import (
    Chunker,
    ChunkerError,
    InvalidChunkSizeError,
    RecursiveChunker,
)
from knowledge_agent.ingestion.loader import PdfLoader
from knowledge_agent.schemas.document import DocumentPage


def page(content: str, page_number: int = 1) -> DocumentPage:
    return DocumentPage(
        document_id="doc",
        filename="doc.pdf",
        page_number=page_number,
        content=content,
    )


def test_chunker_is_abstract():
    with pytest.raises(TypeError):
        Chunker()


def test_size_must_be_positive():
    with pytest.raises(InvalidChunkSizeError, match="chunk_size must be positive"):
        RecursiveChunker(chunk_size=0)


def test_negative_size_is_rejected():
    with pytest.raises(InvalidChunkSizeError):
        RecursiveChunker(chunk_size=-10)


def test_negative_overlap_is_rejected():
    with pytest.raises(InvalidChunkSizeError, match="cannot be negative"):
        RecursiveChunker(chunk_size=100, chunk_overlap=-1)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(InvalidChunkSizeError, match="must be smaller"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_overlap_equal_to_size_is_rejected():
    with pytest.raises(InvalidChunkSizeError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_size_error_is_a_chunker_error():
    with pytest.raises(ChunkerError):
        RecursiveChunker(chunk_size=0)


def test_empty_text_yields_no_chunks():
    assert RecursiveChunker(chunk_size=100).split_text("") == []


def test_whitespace_only_text_yields_no_chunks():
    assert RecursiveChunker(chunk_size=100).split_text("   \n\n \t ") == []


def test_overlap_defaults_to_a_fraction_of_chunk_size():
    assert RecursiveChunker().chunk_overlap == 200
    assert RecursiveChunker(chunk_size=100).chunk_overlap == 20
    assert RecursiveChunker(chunk_size=10).chunk_overlap == 2
    assert RecursiveChunker(chunk_size=1).chunk_overlap == 0


def test_small_chunk_size_works_without_explicit_overlap():
    chunks = RecursiveChunker(chunk_size=10).split_text("aaa bbb ccc ddd")

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_short_text_yields_single_chunk():
    assert RecursiveChunker(chunk_size=100).split_text("hello world") == ["hello world"]


def test_text_is_stripped():
    assert RecursiveChunker(chunk_size=100).split_text("\n\n  hello  \n\n") == ["hello"]


def test_splits_on_paragraph_boundary():
    text = "para one here\n\npara two here"

    assert RecursiveChunker(chunk_size=20, chunk_overlap=0).split_text(text) == [
        "para one here",
        "para two here",
    ]


def test_splits_on_sentence_boundary():
    text = "Sentence one. Sentence two."

    assert RecursiveChunker(chunk_size=18, chunk_overlap=0).split_text(text) == [
        "Sentence one",
        "Sentence two.",
    ]


def test_splits_on_word_boundary():
    text = "aaa bbb ccc ddd"

    assert RecursiveChunker(chunk_size=10, chunk_overlap=0).split_text(text) == [
        "aaa bbb",
        "ccc ddd",
    ]


def test_hard_split_when_no_separator_present():
    chunks = RecursiveChunker(chunk_size=10, chunk_overlap=0).split_text("x" * 25)

    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_overlap_shares_text_between_consecutive_chunks():
    chunks = RecursiveChunker(chunk_size=10, chunk_overlap=4).split_text("aaa bbb ccc ddd")

    assert chunks == ["aaa bbb", "bbb ccc", "ccc ddd"]


def test_no_overlap_when_overlap_is_zero():
    chunks = RecursiveChunker(chunk_size=10, chunk_overlap=0).split_text("aaa bbb ccc ddd")

    assert chunks == ["aaa bbb", "ccc ddd"]


def test_chunk_carries_provenance():
    chunks = RecursiveChunker(chunk_size=10, chunk_overlap=0).chunk(
        page("aaa bbb ccc ddd", page_number=7)
    )

    assert len(chunks) == 2
    assert all(c.document_id == "doc" for c in chunks)
    assert all(c.filename == "doc.pdf" for c in chunks)
    assert all(c.page_number == 7 for c in chunks)
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_chunk_index_restarts_for_each_page():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
    content = "aaa bbb ccc ddd"

    first = chunker.chunk(page(content, page_number=1))
    second = chunker.chunk(page(content, page_number=2))

    assert [c.chunk_index for c in first] == [0, 1]
    assert [c.chunk_index for c in second] == [0, 1]
    assert [c.page_number for c in second] == [2, 2]


def test_empty_page_yields_no_chunks():
    assert RecursiveChunker(chunk_size=100).chunk(page("   ")) == []


def test_chunk_pages_aggregates_across_pages():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
    pages = [page("aaa bbb ccc ddd", 1), page("eee fff ggg hhh", 2)]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) == 4
    assert [c.page_number for c in chunks] == [1, 1, 2, 2]
    assert [c.chunk_index for c in chunks] == [0, 1, 0, 1]


def test_chunk_pages_on_empty_input():
    assert RecursiveChunker(chunk_size=10).chunk_pages([]) == []


def test_no_whitespace_only_chunks():
    chunks = RecursiveChunker(chunk_size=12, chunk_overlap=2).split_text(
        "alpha beta\n\n\ngamma delta epsilon zeta"
    )

    assert chunks
    assert all(chunk.strip() for chunk in chunks)


@pytest.mark.parametrize("size", [1, 5, 10, 40, 200])
def test_no_chunk_ever_exceeds_size(size):
    chunker = RecursiveChunker(chunk_size=size, chunk_overlap=min(size - 1, 7))
    text = "word " * 200

    assert all(len(chunk) <= size for chunk in chunker.split_text(text))


def test_no_unique_word_content_is_lost():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]

    emitted = " ".join(chunker.split_text(" ".join(words))).split()

    for word in words:
        assert word in emitted


@pytest.mark.parametrize("overlap", [0, 1, 3, 7, 19])
def test_size_invariant_holds_over_random_input(overlap):
    random.seed(1234)
    alphabet = string.ascii_lowercase + " \n\n. "
    size = 20

    for _ in range(200):
        text = "".join(random.choice(alphabet) for _ in range(random.randint(0, 300)))
        chunks = RecursiveChunker(chunk_size=size, chunk_overlap=overlap).split_text(text)

        assert all(len(chunk) <= size for chunk in chunks)
        assert all(chunk.strip() for chunk in chunks if text.strip())


def test_custom_chunker_can_subclass_base():
    class WholePageChunker(Chunker):
        def chunk(self, page: DocumentPage) -> list:
            from knowledge_agent.schemas.chunk import Chunk

            if not page.content.strip():
                return []

            return [
                Chunk(
                    document_id=page.document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=0,
                    content=page.content,
                )
            ]

    chunks = WholePageChunker().chunk(page("one chunk per page"))

    assert len(chunks) == 1
    assert chunks[0].content == "one chunk per page"


def test_load_then_chunk_pipeline(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    for text in ("alpha beta gamma delta", "epsilon zeta eta theta"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()

    pages = PdfLoader().load(pdf_path)
    chunks = RecursiveChunker(chunk_size=15, chunk_overlap=3).chunk_pages(pages)

    assert len(pages) == 2
    assert chunks
    assert all(len(c.content) <= 15 for c in chunks)
    assert {c.document_id for c in chunks} == {"paper"}
    assert {c.page_number for c in chunks} == {1, 2}
