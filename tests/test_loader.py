from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

from knowledge_agent.ingestion.loader import (
    DocumentLoadError,
    DocumentNotFoundError,
    Loader,
    LoaderError,
    PdfLoader,
    UnsupportedFileError,
)


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, kind: str) -> str:
        assert kind == "text"
        return self._text


class FakePdf:
    def __init__(self, page_texts: list[str]) -> None:
        self._pages = [FakePage(text) for text in page_texts]
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.closed = True


class ExplodingPdf:
    closed = False

    def __iter__(self):
        raise RuntimeError("boom")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.closed = True


@pytest.fixture
def loader() -> PdfLoader:
    return PdfLoader()


@pytest.fixture
def existing_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "existing.pdf"
    path.write_bytes(b"%PDF-1.7")
    return path


def patched_open(pages: list[str] | object = None):
    if not isinstance(pages, list):
        return patch("knowledge_agent.ingestion.loader.pymupdf.open", return_value=pages)
    doc = FakePdf(pages)
    return patch("knowledge_agent.ingestion.loader.pymupdf.open", return_value=doc)


def make_pdf(path: Path, texts: list[str]) -> Path:
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_load_returns_page_per_non_empty_page(loader, existing_pdf):
    with patched_open(["first page", "second page"]):
        pages = loader.load(existing_pdf)

    assert [p.content for p in pages] == ["first page", "second page"]


def test_page_numbers_are_one_based(loader, existing_pdf):
    with patched_open(["a", "b"]):
        pages = loader.load(existing_pdf)

    assert [p.page_number for p in pages] == [1, 2]


def test_blank_pages_are_skipped_but_numbering_is_preserved(loader, existing_pdf):
    with patched_open(["first", "   \n\t ", "third"]):
        pages = loader.load(existing_pdf)

    assert [(p.page_number, p.content) for p in pages] == [(1, "first"), (3, "third")]


def test_content_is_stripped(loader, existing_pdf):
    with patched_open(["\n  hello world \n\n"]):
        pages = loader.load(existing_pdf)

    assert pages[0].content == "hello world"


def test_metadata_comes_from_path(loader, tmp_path):
    path = tmp_path / "my-report.pdf"
    path.write_bytes(b"%PDF-1.7")

    with patched_open(["text"]):
        page = loader.load(path)[0]

    assert page.document_id == "my-report"
    assert page.filename == "my-report.pdf"


def test_returns_empty_list_when_no_page_has_text(loader, existing_pdf):
    with patched_open(["", "  "]):
        assert loader.load(existing_pdf) == []


def test_empty_pdf_returns_empty_list(loader, existing_pdf):
    with patched_open([]):
        assert loader.load(existing_pdf) == []


def test_open_receives_the_path_object(loader, existing_pdf):
    with patch("knowledge_agent.ingestion.loader.pymupdf.open") as mock_open:
        mock_open.return_value.__enter__.return_value = iter([])
        loader.load(existing_pdf)

    mock_open.assert_called_once_with(existing_pdf)


def test_load_on_real_document(loader, tmp_path):
    path = make_pdf(tmp_path / "sample.pdf", ["page one", "page two"])

    pages = loader.load(path)

    assert [p.page_number for p in pages] == [1, 2]
    assert [p.document_id for p in pages] == ["sample", "sample"]
    assert [p.filename for p in pages] == ["sample.pdf", "sample.pdf"]
    assert "page one" in pages[0].content
    assert "page two" in pages[1].content


def test_missing_file_raises_document_not_found(loader, tmp_path):
    with pytest.raises(DocumentNotFoundError, match="missing.pdf"):
        loader.load(tmp_path / "missing.pdf")


def test_directory_instead_of_file_raises_document_not_found(loader, tmp_path):
    with pytest.raises(DocumentNotFoundError):
        loader.load(tmp_path)


def test_directory_not_found_error_is_loader_error(loader, tmp_path):
    with pytest.raises(LoaderError):
        loader.load(tmp_path / "missing.pdf")


def test_document_not_found_error_is_catchable_as_file_not_found(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "missing.pdf")


def test_document_not_found_error_is_catchable_as_os_error(loader, tmp_path):
    with pytest.raises(OSError):
        loader.load(tmp_path / "missing.pdf")


def test_load_directory_missing_error_is_catchable_as_file_not_found(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_directory(tmp_path / "nope")


def test_every_loader_error_shares_a_common_base():
    for error in (DocumentNotFoundError, UnsupportedFileError, DocumentLoadError):
        assert issubclass(error, LoaderError)


def test_wrong_extension_raises_unsupported_file_error(loader, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFileError, match="notes.txt"):
        loader.load(path)


def test_uppercase_extension_is_accepted(loader, tmp_path):
    path = tmp_path / "REPORT.PDF"
    path.write_bytes(b"%PDF-1.7")

    with patched_open(["text"]):
        assert len(loader.load(path)) == 1


def test_corrupt_pdf_raises_document_load_error(loader, existing_pdf):
    with patch(
        "knowledge_agent.ingestion.loader.pymupdf.open",
        side_effect=pymupdf.FileDataError("cannot open broken file"),
    ):
        with pytest.raises(DocumentLoadError, match="Failed to load PDF"):
            loader.load(existing_pdf)


def test_page_extraction_error_is_wrapped(loader, existing_pdf):
    with patched_open(ExplodingPdf()):
        with pytest.raises(DocumentLoadError, match="boom"):
            loader.load(existing_pdf)


def test_load_error_preserves_underlying_cause(loader, existing_pdf):
    with patch(
        "knowledge_agent.ingestion.loader.pymupdf.open",
        side_effect=pymupdf.FileDataError("root cause"),
    ):
        with pytest.raises(DocumentLoadError) as info:
            loader.load(existing_pdf)

    assert isinstance(info.value.__cause__, pymupdf.FileDataError)


def test_document_is_closed_after_successful_load(loader, existing_pdf):
    doc = FakePdf(["text"])

    with patched_open(doc):
        loader.load(existing_pdf)

    assert doc.closed is True


def test_document_is_closed_when_extraction_fails(loader, existing_pdf):
    doc = ExplodingPdf()

    with patched_open(doc):
        with pytest.raises(DocumentLoadError):
            loader.load(existing_pdf)

    assert doc.closed is True


def test_supports_only_declared_extensions(loader):
    assert loader.supports(Path("a.pdf")) is True
    assert loader.supports(Path("a.PDF")) is True
    assert loader.supports(Path("a.docx")) is False
    assert loader.supports(Path("a")) is False


def test_load_many_concatenates_pages(loader, tmp_path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"%PDF-1.7")
    second.write_bytes(b"%PDF-1.7")

    docs = [first, second]
    with patch("knowledge_agent.ingestion.loader.pymupdf.open", side_effect=docs):
        with patched_open(["page"]):
            pages = loader.load_many([first, second])

    assert [p.page_number for p in pages] == [1, 1]
    assert [p.document_id for p in pages] == ["a", "b"]


def test_load_many_fails_fast_on_bad_file(loader, tmp_path):
    good = tmp_path / "a.pdf"
    good.write_bytes(b"%PDF-1.7")

    with patched_open(["page"]):
        with pytest.raises(DocumentNotFoundError):
            loader.load_many([good, tmp_path / "missing.pdf"])


def test_load_directory_picks_supported_files_only(loader, tmp_path):
    make_pdf(tmp_path / "b.pdf", ["from b"])
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    with patch("knowledge_agent.ingestion.loader.pymupdf.open", return_value=FakePdf(["stub"])):
        paths = loader.load_directory(tmp_path)

    assert {p.filename for p in paths} == {"a.pdf", "b.pdf"}


def test_load_directory_is_sorted(loader, tmp_path):
    for name in ("c.pdf", "a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(b"%PDF-1.7")

    with patch("knowledge_agent.ingestion.loader.pymupdf.open", return_value=FakePdf(["stub"])):
        paths = loader.load_directory(tmp_path)

    assert [p.filename for p in paths] == ["a.pdf", "b.pdf", "c.pdf"]


def test_load_directory_ignores_subdirectories(loader, tmp_path):
    (tmp_path / "sub.pdf").mkdir()

    assert loader.load_directory(tmp_path) == []


def test_load_directory_missing_raises(loader, tmp_path):
    with pytest.raises(DocumentNotFoundError, match="nope"):
        loader.load_directory(tmp_path / "nope")


def test_loader_is_abstract():
    with pytest.raises(TypeError):
        Loader()


def test_custom_loader_can_subclass_base(tmp_path):
    class TxtLoader(Loader):
        extensions = frozenset({".txt"})

        def load(self, path: Path) -> list:
            from knowledge_agent.schemas.document import DocumentPage

            return [
                DocumentPage(
                    document_id=path.stem,
                    filename=path.name,
                    page_number=1,
                    content=path.read_text(encoding="utf-8").strip(),
                )
            ]

    path = tmp_path / "notes.txt"
    path.write_text("  plain text  ", encoding="utf-8")

    pages = TxtLoader().load(path)

    assert pages[0].content == "plain text"
    assert pages[0].document_id == "notes"
