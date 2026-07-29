import fitz  # PyMuPDF
from pathlib import Path


class PDFParser:
    """
    Handles PDF text extraction using PyMuPDF.
    """

    def __init__(self):
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text.
        """
        lines = text.splitlines()

        cleaned_lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        cleaned_text = " ".join(cleaned_lines)

        # Remove excessive whitespace
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text

    def extract_text(self, pdf_path: str):
        """
        Extract text page by page.

        Returns:
        [
            {
                "page": 1,
                "text": "...",
                "source": "sample.pdf"
            }
        ]
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} does not exist.")

        document = fitz.open(pdf_path)

        extracted_pages = []

        for page_number in range(len(document)):
            page = document.load_page(page_number)

            text = page.get_text()

            cleaned = self.clean_text(text)

            if cleaned:
                extracted_pages.append(
                    {
                        "page": page_number + 1,
                        "text": cleaned,
                        "source": pdf_path.name
                    }
                )

        document.close()

        return extracted_pages