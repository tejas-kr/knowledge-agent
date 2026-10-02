from pydantic import BaseModel


class Chunk(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.page_number}:{self.chunk_index}"
