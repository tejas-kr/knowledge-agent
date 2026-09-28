from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import chromadb

from knowledge_agent.config import CHROMA_DIR
from knowledge_agent.embeddings.provider import EmbeddingProvider
from knowledge_agent.schemas.chunk import Chunk
from knowledge_agent.schemas.retrieval import SearchResult

DEFAULT_COLLECTION_NAME = "knowledge"
DEFAULT_TOP_K = 5
SPACE_KEY = "hnsw:space"
COSINE_SPACE = "cosine"


class VectorStoreError(Exception):
    pass


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        provider: EmbeddingProvider,
        directory: Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client=None,
    ) -> None:
        if not collection_name.strip():
            raise VectorStoreError("collection_name cannot be empty")

        self.provider = provider
        self.directory = Path(directory) if directory is not None else CHROMA_DIR
        self.collection_name = collection_name

        self._client = (
            client
            if client is not None
            else chromadb.PersistentClient(path=str(self.directory))
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={SPACE_KEY: COSINE_SPACE},
        )

    @staticmethod
    def chunk_id(chunk: Chunk) -> str:
        return f"{chunk.document_id}:{chunk.page_number}:{chunk.chunk_index}"

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        items = list(chunks)

        if not items:
            return []

        if embeddings is None:
            vectors = self.provider.embed_documents([c.content for c in items])
        else:
            vectors = [list(vector) for vector in embeddings]

            if len(vectors) != len(items):
                raise VectorStoreError(
                    f"Expected {len(items)} embeddings, got {len(vectors)}"
                )

        ids = [self.chunk_id(c) for c in items]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=[c.content for c in items],
                metadatas=[self._metadata(c) for c in items],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store {len(items)} chunks in '{self.collection_name}': {exc}"
            ) from exc

        return ids

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise VectorStoreError("Query cannot be empty")

        if k <= 0:
            raise VectorStoreError(f"k must be positive, got {k}")

        if self.count() == 0:
            return []

        vector = self.provider.embed_query(query)

        try:
            response = self._collection.query(
                query_embeddings=[vector],
                n_results=k,
                where=where or None,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Search failed in '{self.collection_name}': {exc}"
            ) from exc

        return self._to_results(response)

    def delete_document(self, document_id: str) -> None:
        if not document_id.strip():
            raise VectorStoreError("document_id cannot be empty")

        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete document '{document_id}': {exc}"
            ) from exc

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to count chunks in '{self.collection_name}': {exc}"
            ) from exc

    def _metadata(self, chunk: Chunk) -> dict:
        return {
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }

    def _to_results(self, response: dict) -> list[SearchResult]:
        ids = self._first(response, "ids")
        documents = self._first(response, "documents")
        metadatas = self._first(response, "metadatas")
        distances = self._first(response, "distances")

        results: list[SearchResult] = []

        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None
            metadata = metadata or {}

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=documents[index] if index < len(documents) else "",
                    distance=distances[index] if index < len(distances) else 0.0,
                    document_id=metadata.get("document_id", ""),
                    filename=metadata.get("filename", ""),
                    page_number=metadata.get("page_number", 0),
                    chunk_index=metadata.get("chunk_index", 0),
                )
            )

        return results

    @staticmethod
    def _first(response: dict, key: str) -> list:
        value = response.get(key) or []
        return value[0] if value else []
