from pydantic import BaseModel


class DocumentPage(BaseModel):
    document_id: str
    filename: str
    page_number: int
    content: str
    