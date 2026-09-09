from backend.pdf_parser import PDFParser
from backend.utils import TextChunker
from backend.embeddings import EmbeddingGenerator
from backend.database import ChromaDBManager


parser = PDFParser()
chunker = TextChunker()
embedder = EmbeddingGenerator()
db = ChromaDBManager()

pages = parser.extract_text("uploads/Sample.pdf")
chunks = chunker.chunk_pages(pages)
embeddings = embedder.embed_chunks(chunks)

document_id = db.add_chunks(chunks, embeddings)

print()
print("Upload Successful!")
print(f"Document ID: {document_id}")