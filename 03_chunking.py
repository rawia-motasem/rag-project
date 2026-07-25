"""
03_chunking.py
---------------
Step 3 of the RAG pipeline: Chunking.
Splits each preprocessed document into smaller overlapping chunks.
"""

import importlib

preprocessing_module = importlib.import_module("02_preprocessing")


def chunk_text(text, chunk_size=150, overlap=30):
    """Splits text into overlapping chunks of a given size (in characters)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_documents(documents):
    """Splits every document's text into chunks, keeping metadata attached."""
    all_chunks = []
    for doc in documents:
        text_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(text_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "status": doc["status"],
                    "department": doc["department"],
                    "text": chunk,
                }
            )
    return all_chunks


def get_chunked_documents():
    """Convenience function: loads preprocessed docs and returns them chunked."""
    preprocessed_docs = preprocessing_module.get_preprocessed_documents()
    return chunk_documents(preprocessed_docs)


if __name__ == "__main__":
    chunked_docs = get_chunked_documents()
    print(f"Created {len(chunked_docs)} chunks:\n")
    for c in chunked_docs:
        print(f"- [{c['chunk_id']}] {c['text'][:60]}...")
