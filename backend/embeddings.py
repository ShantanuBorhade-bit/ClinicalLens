from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_chunks(self, chunks) -> list:
        """
        Generate embeddings for all chunks.
        """
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings