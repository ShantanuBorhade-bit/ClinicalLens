from backend.pdf_parser import PDFParser

parser = PDFParser()

pages = parser.extract_text("uploads/Sample.pdf")

print(f"Pages extracted: {len(pages)}")

for page in pages:
    print("-" * 60)
    print(f"Page: {page.page}")
    print(page.text[:300])