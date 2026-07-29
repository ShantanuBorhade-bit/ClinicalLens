from backend.database import ChromaDBManager

db = ChromaDBManager()

collection = db.get_collection()

print(collection.name)