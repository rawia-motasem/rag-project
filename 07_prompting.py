"""
07_prompting.py
-----------------
Step 7 of the RAG pipeline: Prompting.
Builds a prompt from the retrieved context and calls the LLM (via OpenRouter)
to produce a final, source-grounded answer.

IMPORTANT (API key rules):
    - No real API key is written in this file.
    - The key is read from an environment variable OPENROUTER_API_KEY.
    - streamlit_app.py sets this from Streamlit secrets before calling generate_answer().
"""

import os
import importlib
from openai import OpenAI

retrieve_module = importlib.import_module("06_retrieve_context")

# These are read from the environment. streamlit_app.py overwrites them
# with values from st.secrets when running on Streamlit Cloud.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_client = None


def get_client():
    """Creates (or returns) the OpenAI-compatible client pointed at OpenRouter."""
    global _client
    if _client is None or _client.api_key != OPENROUTER_API_KEY:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client


def build_prompt(query, retrieved_chunks):
    """Builds a prompt that instructs the model to answer using only the retrieved context."""
    context_text = "\n\n".join(
        f"[Source: {c['metadata']['title']} | Status: {c['metadata']['status']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""Answer the question using ONLY the context below. If the context does not contain the answer, say you don't know.
Always mention which source(s) you used in your answer.

Context:
{context_text}

Question: {query}

Answer:"""
    return prompt

def generate_rag_response(query, n_results=3):
    """Full pipeline: retrieve context, build prompt, call the LLM, return the answer."""
    retrieved_chunks = retrieve_module.retrieve_context(query, n_results=n_results)
    prompt = build_prompt(query, retrieved_chunks)

    client = get_client()
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    return answer, retrieved_chunks


if __name__ == "__main__":
    test_query = "By what percentage must greenhouse gases drop by 2030?"
    answer, sources = generate_answer(test_query)

    print("Question:", test_query)
    print("\nAnswer:", answer)
    print("\nSources used:")
    for s in sources:
        print(f"- {s['metadata']['title']} ({s['metadata']['status']})")
