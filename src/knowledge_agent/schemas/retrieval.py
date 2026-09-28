from pydantic import BaseModel


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    distance: float
    document_id: str
    filename: str
    page_number: int
    chunk_index: int

    @property
    def similarity(self) -> float:
        return 1 - self.distance
