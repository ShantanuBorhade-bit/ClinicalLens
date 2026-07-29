from backend.database import ChromaDBManager

db = ChromaDBManager()

print("\nUploaded Documents\n")

documents = db.list_documents()

if not documents:
    print("No documents found.")

else:
    for doc_id, filename in documents.items():
        print(f"{filename}")
        print(doc_id)
        print("-" * 60)