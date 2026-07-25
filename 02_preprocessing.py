"""
02_preprocessing.py
--------------------
Step 2 of the RAG pipeline: Preprocessing.
Cleans raw document text (extra whitespace, blank lines) before chunking.
"""

import re
import importlib

documents_module = importlib.import_module("01_documents")


def clean_text(text):
    """Cleans a single piece of text: removes extra whitespace and blank lines."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_documents(documents):
    """Takes the raw documents list and returns a new list with cleaned text."""
    cleaned_docs = []
    for doc in documents:
        cleaned_doc = doc.copy()
        cleaned_doc["text"] = clean_text(doc["text"])
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs


def get_preprocessed_documents():
    """Convenience function: loads raw documents and returns them cleaned."""
    raw_docs = documents_module.get_raw_documents()
    return preprocess_documents(raw_docs)


if __name__ == "__main__":
    preprocessed_docs = get_preprocessed_documents()
    print(f"Preprocessed {len(preprocessed_docs)} documents:\n")
    for d in preprocessed_docs:
        print(f"- [{d['id']}] {d['title']} -> {len(d['text'])} characters")
