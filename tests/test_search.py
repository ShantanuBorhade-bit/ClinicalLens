from backend.embeddings import EmbeddingGenerator
from backend.database import ChromaDBManager


embedder = EmbeddingGenerator()
db = ChromaDBManager()

query = "What is the conclusion of the clinical study?"

query_embedding = embedder.embed_text(query)

results = db.search(query_embedding)

print("\nTop Results\n")

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, (doc, meta, dist) in enumerate(
    zip(documents, metadatas, distances),
    start=1
):
    print("=" * 80)
    print(f"Result {i}")
    print(f"Distance : {dist:.4f}")
    print(f"Page     : {meta['page']}")
    print(f"File     : {meta['filename']}")
    print(f"Document : {meta['document_id']}")
    print()
    print(doc[:500])
    print()