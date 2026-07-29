from backend.pdf_parser import PDFParser
from backend.utils import TextChunker
from backend.embeddings import EmbeddingGenerator

parser = PDFParser()
pages = parser.extract_text("uploads/Sample.pdf")   # Use your actual filename

chunker = TextChunker()
chunks = chunker.chunk_pages(pages)

embedder = EmbeddingGenerator()

vectors = embedder.embed_chunks(chunks)

print(f"\nGenerated {len(vectors)} embeddings.")
print(f"Embedding Dimension: {len(vectors[0])}")