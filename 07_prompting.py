import os
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="RAG Climate Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 RAG Climate Intelligence & Search Engine")
st.markdown(
    "Search climate policies using Hybrid Retrieval (BM25 + Dense Embeddings) & LLM Analysis."
)


# --- 2. Load Knowledge Base & Process Chunks ---
@st.cache_resource
def initialize_rag_system():
    files_data = {
        "paris_agreement_overview.txt": """The Paris Agreement is a legally binding international treaty on climate change. Its goal is to limit global warming to well below 2, preferably to 1.5 degrees Celsius.""",
        "climate_economy.txt": """Transitioning to a green economy opens up vast profitable opportunities in zero-carbon business. Investment in renewable energy, electric infrastructure, and sustainable agriculture drives long-term economic growth.""",
        "cop_agreements.txt": """The COP agreements establish a robust global framework that accelerates green tech sharing and knowledge transfer. This framework ensures that developing nations gain immediate access to advanced sustainable technologies.""",
        "old_emission_projections.txt": """[OUTDATED REPORT 2015] Initial emissions projections suggested a slower decline curve. This historical data is no longer accurate for current 2030 policy planning.""",
        "old_finance_notice.txt": """[OUTDATED NOTICE 2020] The old 2020 target for climate finance aimed to mobilize 100 billion dollars annually from developed countries. This target has been updated in subsequent sessions.""",
    }

    documents = []
    for doc_id, (filename, content) in enumerate(files_data.items()):
        is_current = False if "old" in filename else True
        doc_type = "old notice" if "old" in filename else "policy"
        documents.append({
            "document_id": doc_id,
            "title": filename.replace(".txt", "").replace("_", " ").title(),
            "department": (
                "Climate Action" if is_current else "Historical Records"
            ),
            "doc_type": doc_type,
            "effective_date": "2025-09-01" if is_current else "2019-01-01",
            "is_current": is_current,
            "text": content,
        })

    # Chunking Function
    def chunk_text(text, chunk_size=25, overlap=5):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += chunk_size - overlap
        return chunks

    chunk_rows = []
    for doc in documents:
        for chunk_index, chunk_text_value in enumerate(
            chunk_text(doc["text"])
        ):
            search_text = f"{doc['title']} {doc['department']} {doc['doc_type']} {chunk_text_value}"
            chunk_rows.append({
                "chunk_id": f"{doc['document_id']}_chunk{chunk_index}",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "department": doc["department"],
                "doc_type": doc["doc_type"],
                "effective_date": doc["effective_date"],
                "is_current": doc["is_current"],
                "chunk_text": chunk_text_value,
                "search_text": search_text,
            })

    chunks_df = pd.DataFrame(chunk_rows)

    # Initialize Models
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(
        chunks_df["search_text"].tolist(), normalize_embeddings=True
    )

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        chunks_df["search_text"].apply(
            lambda x: re.sub(
                r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", x.lower())
            ).strip()
        )
    )

    tokenized_chunks = [
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", text.lower()))
        .strip()
        .split()
        for text in chunks_df["search_text"]
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    return chunks_df, model, chunk_embeddings, tfidf_vectorizer, tfidf_matrix, bm25


chunks_df, model, chunk_embeddings, tfidf_vectorizer, tfidf_matrix, bm25 = (
    initialize_rag_system()
)

# --- 3. UI Controls ---
st.sidebar.header("⚙️ Search Settings")
retrieval_method = st.sidebar.selectbox(
    "Choose Retrieval Strategy", ["Hybrid (Recommended)", "BM25", "Embeddings"]
)
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=5, value=3)

query = st.text_input(
    "🔍 Enter your question:",
    "Where can poor countries get monetary aid for climate change?",
)


# --- 4. Retrieval Functions ---
def min_max_normalize(scores):
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


if st.button("Search & Analyze"):
    with st.spinner("Searching knowledge base..."):
        norm_query = re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", query.lower())
        ).strip()

        if retrieval_method == "BM25":
            scores = bm25.get_scores(norm_query.split())
            ranking = np.argsort(scores)[::-1][:top_k]
            res = chunks_df.iloc[ranking].copy()
            res["score"] = scores[ranking]

        elif retrieval_method == "Embeddings":
            q_embed = model.encode([query], normalize_embeddings=True)
            scores = cosine_similarity(q_embed, chunk_embeddings).flatten()
            ranking = np.argsort(scores)[::-1][:top_k]
            res = chunks_df.iloc[ranking].copy()
            res["score"] = scores[ranking]

        else:  # Hybrid
            lex_scores = cosine_similarity(
                tfidf_vectorizer.transform([norm_query]), tfidf_matrix
            ).flatten()
            sem_scores = cosine_similarity(
                model.encode([query], normalize_embeddings=True),
                chunk_embeddings,
            ).flatten()
            combined_scores = 0.6 * min_max_normalize(
                sem_scores
            ) + 0.4 * min_max_normalize(lex_scores)
            ranking = np.argsort(combined_scores)[::-1][:top_k]
            res = chunks_df.iloc[ranking].copy()
            res["score"] = combined_scores[ranking]

        st.subheader("📌 Retrieved Document Chunks")
        for idx, row in res.iterrows():
            with st.expander(
                f"📄 {row['title']} (Score: {row['score']:.4f}) - [{row['department']}]"
            ):
                st.write(f"**Chunk Text:** {row['chunk_text']}")
                st.caption(
                    f"Doc Type: {row['doc_type']} | Effective Date: {row['effective_date']} | Is Current: {row['is_current']}"
                )
