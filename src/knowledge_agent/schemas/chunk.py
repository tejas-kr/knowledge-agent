from pydantic import BaseModel


class Chunk(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
