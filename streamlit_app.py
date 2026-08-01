"""
streamlit_app.py
------------------
Streamlit UI for the "Green Future" Climate RAG assistant.
Ties together documents -> preprocessing -> chunking -> vector representation
-> vector store -> retrieval -> prompting, and shows an answer with sources.
"""

import importlib
import streamlit as st

rag = importlib.import_module("07_prompting")

st.set_page_config(page_title="Green Future | Climate RAG Assistant", page_icon="🌿", layout="centered")

# ---- Load the API key from Streamlit secrets (never hard-coded) ----
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Visual identity: "Green Future"
# Palette   — background #10241C (deep forest), surface #16332A, text #F3F1E9,
#             accent #D9A441 (amber, warmth of sunlight through leaves),
#             sage #8FAE8B (secondary / muted).
# Type      — Fraunces (display serif, headings) + Inter (body / UI).
# Signature — a single growth-ring arc behind the title, echoing tree rings /
#             cyclical climate data, used once and never repeated.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #10241C;
        color: #F3F1E9;
    }

    .gf-hero {
        position: relative;
        padding: 2.2rem 0 1.4rem 0;
        margin-bottom: 0.6rem;
    }
    .gf-ring {
        position: absolute;
        top: -30px;
        left: -40px;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        border: 1px solid rgba(217, 164, 65, 0.25);
        z-index: 0;
    }
    .gf-ring::before {
        content: "";
        position: absolute;
        top: 22px; left: 22px;
        width: 116px; height: 116px;
        border-radius: 50%;
        border: 1px solid rgba(143, 174, 139, 0.25);
    }
    .gf-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8FAE8B;
        margin-bottom: 0.4rem;
        position: relative;
        z-index: 1;
    }
    .gf-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.6rem;
        line-height: 1.05;
        color: #F3F1E9;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .gf-title span { color: #D9A441; }
    .gf-subtitle {
        font-size: 0.98rem;
        color: #C9C7BC;
        margin-top: 0.6rem;
        max-width: 34rem;
        position: relative;
        z-index: 1;
    }

    div[data-testid="stButton"] button {
        background: #16332A;
        color: #F3F1E9;
        border: 1px solid rgba(143, 174, 139, 0.35);
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #D9A441;
        color: #D9A441;
    }

    .stTextInput input {
        background: #16332A;
        color: #F3F1E9;
        border: 1px solid rgba(243, 241, 233, 0.15);
        border-radius: 10px;
    }

    .gf-card {
        background: #16332A;
        border-left: 3px solid #D9A441;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .gf-card-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8FAE8B;
        margin-bottom: 0.5rem;
    }
    .gf-source-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(243, 241, 233, 0.08);
        font-size: 0.92rem;
    }
    .gf-source-row:last-child { border-bottom: none; }
    .gf-status-current {
        color: #8FAE8B;
        font-size: 0.75rem;
        border: 1px solid rgba(143, 174, 139, 0.4);
        border-radius: 999px;
        padding: 0.05rem 0.5rem;
        margin-left: 0.4rem;
    }
    .gf-status-outdated {
        color: #D98741;
        font-size: 0.75rem;
        border: 1px solid rgba(217, 135, 65, 0.4);
        border-radius: 999px;
        padding: 0.05rem 0.5rem;
        margin-left: 0.4rem;
    }

    section[data-testid="stSidebar"] {
        background: #0C1C16;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gf-hero">
        <div class="gf-ring"></div>
        <div class="gf-eyebrow">Climate Policy · Retrieval-Augmented Generation</div>
        <div class="gf-title">Green <span>Future</span></div>
        <div class="gf-subtitle">
            Ask a question about climate policy and receive an answer grounded
            entirely in retrieved source documents — every claim traced back
            to where it came from.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "This assistant retrieves relevant passages from a small set of "
        "climate-policy documents and asks an LLM to answer using only "
        "that retrieved context — no outside knowledge."
    )
    n_results = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)
    st.markdown("---")
    st.caption("Built with Streamlit · Chroma · Sentence-Transformers · OpenRouter")

SAMPLE_QUESTIONS = [
    "By what percentage must greenhouse gases drop by 2030?",
    "What replaced the old climate finance target?",
    "How has battery storage improved renewable energy?",
    "What is the main goal of the Paris Agreement?",
    "Which sectors are attracting the most green investment?",
    "What role does the COP play in the UN climate framework?",
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

st.markdown("<div class='gf-card-label' style='margin-bottom:0.4rem;'>Try asking</div>", unsafe_allow_html=True)

ROW_SIZE = 3
for row_start in range(0, len(SAMPLE_QUESTIONS), ROW_SIZE):
    row_questions = SAMPLE_QUESTIONS[row_start: row_start + ROW_SIZE]
    chip_cols = st.columns(ROW_SIZE)
    for col, q in zip(chip_cols, row_questions):
        with col:
            if st.button(q, key=f"chip_{q}"):
                st.session_state.query_input = q

query = st.text_input(
    "Your question",
    value=st.session_state.query_input,
    placeholder="e.g. By what percentage must greenhouse gases drop by 2030?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query:
    if not rag.OPENROUTER_API_KEY:
        st.error("No API key found. Please configure OPENROUTER_API_KEY in Streamlit secrets.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            answer, sources = rag.generate_answer(query, n_results=n_results)

        st.markdown(
            f"""
            <div class="gf-card">
                <div class="gf-card-label">Answer</div>
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_rows = ""
        for s in sources:
            status_class = "gf-status-current" if s["metadata"]["status"] == "CURRENT" else "gf-status-outdated"
            source_rows += f"""
            <div class="gf-source-row">
                <span>{s['metadata']['title']} <span class="{status_class}">{s['metadata']['status']}</span></span>
                <span style="color:#8FAE8B;">distance {s['distance']:.4f}</span>
            </div>
            """

        st.markdown(
            f"""
            <div class="gf-card">
                <div class="gf-card-label">Sources used</div>
                {source_rows}
            </div>
            """,
            unsafe_allow_html=True,
        )
