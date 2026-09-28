from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import pymupdf

from knowledge_agent.schemas.document import DocumentPage


class LoaderError(Exception):
    pass


class DocumentNotFoundError(LoaderError, FileNotFoundError):
    pass


class UnsupportedFileError(LoaderError):
    pass


class DocumentLoadError(LoaderError):
    pass


class Loader(ABC):
    extensions: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def load(self, path: Path) -> list[DocumentPage]:
        raise NotImplementedError

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def load_many(self, paths: Iterable[Path]) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        for path in paths:
            pages.extend(self.load(path))
        return pages

    def load_directory(self, directory: Path) -> list[DocumentPage]:
        if not directory.is_dir():
            raise DocumentNotFoundError(f"Directory not found: {directory}")

        paths = sorted(p for p in directory.iterdir() if p.is_file() and self.supports(p))
        return self.load_many(paths)


class PdfLoader(Loader):
    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})

    def load(self, path: Path) -> list[DocumentPage]:
        if not path.is_file():
            raise DocumentNotFoundError(f"Document not found: {path}")

        if not self.supports(path):
            raise UnsupportedFileError(
                f"{self.__class__.__name__} cannot load '{path.name}': "
                f"expected one of {sorted(self.extensions)}"
            )

        try:
            with pymupdf.open(path) as pdf:
                return self._extract(path, pdf)
        except LoaderError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Failed to load PDF '{path}': {exc}") from exc

    def _extract(self, path: Path, pdf) -> list[DocumentPage]:
        pages: list[DocumentPage] = []

        for page_number, page in enumerate(pdf, start=1):
            content = page.get_text("text").strip()

            if not content:
                continue

            pages.append(
                DocumentPage(
                    document_id=path.stem,
                    filename=path.name,
                    page_number=page_number,
                    content=content,
                )
            )

        return pages
