from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

from knowledge_agent.schemas.chunk import Chunk
from knowledge_agent.schemas.document import DocumentPage


class ChunkerError(Exception):
    pass


class InvalidChunkSizeError(ChunkerError):
    pass


class Chunker(ABC):
    @abstractmethod
    def chunk(self, page: DocumentPage) -> list[Chunk]:
        raise NotImplementedError

    def chunk_pages(self, pages: Iterable[DocumentPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self.chunk(page))
        return chunks


class RecursiveChunker(Chunker):
    separators: ClassVar[tuple[str, ...]] = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int | None = None) -> None:
        if chunk_size <= 0:
            raise InvalidChunkSizeError(f"chunk_size must be positive, got {chunk_size}")

        if chunk_overlap is None:
            chunk_overlap = min(chunk_size * 2 // 10, chunk_size - 1)

        if chunk_overlap < 0:
            raise InvalidChunkSizeError(
                f"chunk_overlap cannot be negative, got {chunk_overlap}"
            )

        if chunk_overlap >= chunk_size:
            raise InvalidChunkSizeError(
                f"chunk_overlap ({chunk_overlap}) must be smaller "
                f"than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, page: DocumentPage) -> list[Chunk]:
        return [
            Chunk(
                document_id=page.document_id,
                filename=page.filename,
                page_number=page.page_number,
                chunk_index=index,
                content=text,
            )
            for index, text in enumerate(self.split_text(page.content))
        ]

    def split_text(self, text: str) -> list[str]:
        return self._split(text.strip(), self.separators)

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        for index, separator in enumerate(separators):
            if separator in text:
                break
        else:
            return self._hard_split(text)

        remaining = separators[index + 1:]

        chunks: list[str] = []
        buffer = ""

        for piece in (p for p in text.split(separator) if p.strip()):
            candidate = f"{buffer}{separator}{piece}" if buffer else piece

            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = self._overlap_tail(buffer)

            if len(piece) <= self.chunk_size:
                seeded = f"{buffer}{separator}{piece}" if buffer else piece
                buffer = seeded if len(seeded) <= self.chunk_size else piece
            else:
                chunks.extend(self._split(piece, remaining))
                buffer = ""

        if buffer:
            chunks.append(buffer)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap

        chunks: list[str] = []
        start = 0

        while start < len(text):
            chunks.append(text[start : start + self.chunk_size])
            start += step

        return chunks

    def _overlap_tail(self, text: str) -> str:
        if self.chunk_overlap <= 0:
            return ""

        tail = text[-self.chunk_overlap :]
        boundary = tail.rfind(" ")

        if boundary >= 0:
            tail = tail[boundary + 1 :]

        return tail


if __name__ == "__main__":

    from knowledge_agent.ingestion.loader import PdfLoader
    from pathlib import Path


    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    pdf_text = PdfLoader().load(path=Path("documents/1706.03762v7.pdf"))
    print(pdf_text)
    chunks = chunker.split_text(pdf_text[0].content)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")
