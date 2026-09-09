import uuid

import chromadb
from chromadb.config import Settings

from backend.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME
)


class ChromaDBManager:
    """
    Handles all vector database operations.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def get_collection(self):
        """
        Returns the ChromaDB collection.
        """
        return self.collection

    def add_chunks(self, chunks, embeddings):
        """
        Stores chunks and embeddings in ChromaDB.

        Returns:
            document_id (str)
        """

        document_id = str(uuid.uuid4())

        # ChromaDB rejects empty documents, so drop empty chunks up front.
        pairs = [
            (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
            if chunk.text and chunk.text.strip()
        ]
        if not pairs:
            raise ValueError("No non-empty chunks to store.")
        chunks, embeddings = (list(t) for t in zip(*pairs))

        ids = [
            f"{document_id}_{chunk.chunk_id}"
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": document_id,
                "filename": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Stored {len(chunks)} chunks.")
        print(f"Document ID: {document_id}")

        return document_id

    def search(self, query_embedding, top_k=4):
        """
        Semantic search.
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results

    def list_documents(self):
        """
        Lists all unique uploaded documents.
        """

        data = self.collection.get(
            include=["metadatas"]
        )

        documents = {}

        for metadata in data.get("metadatas") or []:

            doc_id = metadata.get("document_id")
            filename = metadata.get("filename")

            # Skip malformed metadata instead of raising a KeyError.
            if not doc_id:
                continue

            documents[doc_id] = filename or "unknown"

        return documents

    def delete_document_by_filename(self, filename):
        """
        Deletes all chunks whose metadata filename matches. Used when a PDF is
        re-uploaded so old chunks are replaced rather than duplicated.
        """

        self.collection.delete(
            where={
                "filename": filename
            }
        )

        print(f"Deleted existing chunks for file: {filename}")

    def delete_document(self, document_id):
        """
        Deletes all chunks belonging to one document.
        """

        self.collection.delete(
            where={
                "document_id": document_id
            }
        )

        print(f"Deleted document: {document_id}")

    def clear_database(self):
        """
        Deletes every chunk in the collection.
        """

        self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        print("Database cleared.")