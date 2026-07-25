"""
05_create_chroma_store.py
---------------------------
Step 5 of the RAG pipeline: Vector Store.
Stores embedded chunks in a Chroma vector store for similarity search.
"""

import importlib
import chromadb

vector_module = importlib.import_module("04_vector_representation")

_client = None
_collection = None


def get_collection():
    """Creates (or returns) the Chroma collection with all chunks stored."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.Client()

        # Recreate the collection each time this runs, to avoid duplicate-ID errors
        try:
            _client.delete_collection(name="climate_docs")
        except Exception:
            pass

        _collection = _client.create_collection(name="climate_docs")

        embedded_chunks = vector_module.get_embedded_chunks()

        _collection.add(
            ids=[c["chunk_id"] for c in embedded_chunks],
            embeddings=[c["embedding"].tolist() for c in embedded_chunks],
            documents=[c["text"] for c in embedded_chunks],
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "status": c["status"],
                    "department": c["department"],
                }
                for c in embedded_chunks
            ],
        )
    return _collection


if __name__ == "__main__":
    collection = get_collection()
    print(f"Stored {collection.count()} chunks in Chroma vector store")
