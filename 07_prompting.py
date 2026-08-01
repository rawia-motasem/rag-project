"""
07_prompting.py
-----------------
Step 7 of the RAG pipeline: Prompting & Generation.
Retrieves relevant context chunks (step 6) and asks an LLM (via OpenRouter)
to answer using ONLY that context.
"""

import importlib
from openai import OpenAI

retrieve_module = importlib.import_module("06_retrieve_context")


def build_prompt(query, context_chunks):
    context_text = "\n\n".join(
        f"[Source: {c['metadata'].get('title', c['chunk_id'])}]\n{c['text']}"
        for c in context_chunks
    )
    return f"""You are a climate policy assistant. Answer the question using ONLY the
context below. If the answer is not contained in the context, say you don't know.

Context:
{context_text}

Question: {query}

Answer:"""


def generate_rag_response(query, api_key, model="openai/gpt-4o-mini", n_results=3):
    """Retrieves context for the query and generates an answer grounded in it."""
    context_chunks = retrieve_module.retrieve_context(query, n_results=n_results)

    if not context_chunks:
        return "No relevant context was found for this question."

    prompt = build_prompt(query, context_chunks)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    import os
    test_query = "By what percentage must greenhouse gases drop by 2030?"
    answer = generate_rag_response(test_query, api_key=os.getenv("OPENROUTER_API_KEY", ""))
    print(f"Query: {test_query}\n\nAnswer:\n{answer}")
