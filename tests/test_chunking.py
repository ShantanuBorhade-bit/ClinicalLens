from backend.pdf_parser import PDFParser
from backend.utils import TextChunker

parser = PDFParser()
pages = parser.extract_text("uploads/Sample.pdf")

chunker = TextChunker()

chunks = chunker.chunk_pages(pages)

print(f"\nTotal Chunks: {len(chunks)}\n")

for chunk in chunks[:5]:
    print("=" * 80)
    print(chunk.chunk_id)
    print(chunk.page)
    print(chunk.source)
    print(chunk.text[:400])