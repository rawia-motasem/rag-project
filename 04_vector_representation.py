"""
04_vector_representation.py
-----------------------------
Step 4 of the RAG pipeline: Vector Representation.
Converts text chunks into numeric embeddings using sentence-transformers.
"""

import importlib
from sentence_transformers import SentenceTransformer

chunking_module = importlib.import_module("03_chunking")

_model = None


def get_model():
    """Loads the embedding model once and reuses it (avoids reloading)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_chunks(chunks):
    """Adds a vector embedding to each chunk dictionary."""
    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def get_embedded_chunks():
    """Convenience function: loads chunked docs and returns them embedded."""
    chunked_docs = chunking_module.get_chunked_documents()
    return embed_chunks(chunked_docs)


if __name__ == "__main__":
    embedded_chunks = get_embedded_chunks()
    print(f"Embedded {len(embedded_chunks)} chunks")
    print(f"Embedding size (vector dimension): {len(embedded_chunks[0]['embedding'])}")
    print(f"Example vector (first 5 numbers): {embedded_chunks[0]['embedding'][:5]}")
