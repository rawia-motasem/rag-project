"""
streamlit_app.py
------------------
Streamlit UI for the Climate RAG assistant.
Ties together documents -> preprocessing -> chunking -> vector representation
-> vector store -> retrieval -> prompting, and shows an answer with sources.
"""

import importlib
import streamlit as st

rag = importlib.import_module("07_prompting")

st.set_page_config(page_title="Climate RAG Assistant", page_icon="🌍")

# ---- Load the API key from Streamlit secrets (never hard-coded) ----
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

st.title("🌍 Climate RAG Assistant")
st.caption("Ask a question and get an answer grounded in the climate policy documents below.")

with st.sidebar:
    st.subheader("About")
    st.write(
        "This assistant retrieves relevant chunks from a small set of "
        "climate-policy documents and asks an LLM to answer using only "
        "that retrieved context."
    )
    n_results = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)

query = st.text_input("Your question", placeholder="e.g. By what percentage must greenhouse gases drop by 2030?")

if st.button("Ask") and query:
    if not rag.OPENROUTER_API_KEY:
        st.error("No API key found. Please configure OPENROUTER_API_KEY in Streamlit secrets.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            answer, sources = rag.generate_answer(query, n_results=n_results)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources used")
        for s in sources:
            st.write(
                f"- **{s['metadata']['title']}** "
                f"({s['metadata']['status']}, distance: {s['distance']:.4f})"
            )
