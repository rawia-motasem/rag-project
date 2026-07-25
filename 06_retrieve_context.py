"""
06_retrieve_context.py
------------------------
Step 6 of the RAG pipeline: Context Retrieval.
Given a user query, finds the most relevant chunks from the vector store.
"""

import importlib

vector_module = importlib.import_module("04_vector_representation")
store_module = importlib.import_module("05_create_chroma_store")


def retrieve_context(query, n_results=3):
    """Takes a user query and returns the most relevant chunks from Chroma."""
    model = vector_module.get_model()
    collection = store_module.get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return retrieved


if __name__ == "__main__":
    test_query = "By what percentage must greenhouse gases drop by 2030?"
    results = retrieve_context(test_query)

    print(f"Query: {test_query}\n")
    for r in results:
        print(f"- [{r['chunk_id']}] (distance: {r['distance']:.4f})")
        print(f"  {r['text']}\n")
