from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.models import DocumentChunk

class TextChunker:
    """
    Splits extracted PDF text into overlapping chunks.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def chunk_pages(self, pages):
        """
        Convert pages into chunks.
        """

        chunks = []
        chunk_id = 0

        for page in pages:
            split_text = self.splitter.split_text(page.text)

            for piece in split_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=piece,
                        page=page.page,
                        source=page.source
                    )
                )
                chunk_id += 1

        return chunks