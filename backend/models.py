from dataclasses import dataclass


@dataclass
class DocumentPage:
    """
    Represents one extracted page from a PDF.
    """
    page: int
    text: str
    source: str


@dataclass
class DocumentChunk:
    """
    Represents one searchable chunk.
    """
    chunk_id: int
    text: str
    page: int
    source: str