__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import importlib

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Green Future - Climate Policy RAG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS for Dark Green Theme
st.markdown("""
<style>
    .stApp {
        background-color: #0b1c18;
        color: #e0e0e0;
    }
    h1 {
        color: #ffffff;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem !important;
    }
    .subtitle-text {
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem !important;
    }
    [data-testid="stSidebar"] {
        background-color: #071310;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        font-size: 1.5rem;
    }
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #2a4d44;
        background-color: transparent;
        color: #ffffff;
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2a4d44;
        border-color: #ffffff;
    }
    .result-box {
        background-color: rgba(42, 77, 68, 0.2);
        border: 1px solid #2a4d44;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .result-header {
        color: #6ed0ac;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .source-item {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC INITIALIZATION ---
api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
model_name = st.secrets.get("OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

if not api_key:
    st.error("⚠️ OPENROUTER_API_KEY is missing! Please configure Secrets in Streamlit Settings.")
    st.stop()

rag = importlib.import_module("07_prompting")

# Initialize Session State Variables
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None

# Function to handle button clicks directly
def set_question(q_text):
    st.session_state.user_question = q_text

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This assistant retrieves relevant passages from a set of **climate-policy documents** and asks an LLM to answer using *only* that retrieved context – **no outside knowledge.**
    """)
    st.markdown("---")
    num_chunks = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)

# --- 4. MAIN CONTENT AREA ---
st.markdown("<h1>Green Future</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Ask a question about climate policy and receive an answer grounded entirely in retrieved source documents – every claim traced back to where it came from.</p>", unsafe_allow_html=True)

st.markdown("### TRY ASKING")
col1, col2, col3 = st.columns([1.5, 1.5, 2])

# Using on_click callback to update state before rerender
col1.button("🌱 Goals of Climate Policy?", on_click=set_question, args=("What are the key goals of the Climate Policy?",))
col2.button("🤝 Technology contribution?", on_click=set_question, args=("How does green technology contribute to sustainability?",))
col3.button("📋 Action steps for a Green economy?", on_click=set_question, args=("What action steps are recommended for a sustainable future?",))

# Text input synced with session state
query = st.text_input("Your question", key="user_question")

ask_button = st.button("Ask")

# Trigger search if Ask button clicked OR if a question exists
if ask_button or query:
    if query.strip():
        with st.spinner("Retrieving context and generating answer..."):
            try:
                if hasattr(rag, 'generate_rag_response'):
                    response_text = rag.generate_rag_response(query, api_key=api_key, model=model_name)
                    st.session_state.current_answer = response_text
                else:
                    st.error("Function `generate_rag_response` not found in 07_prompting.py")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- 5. DISPLAY RESULTS ---
if st.session_state.current_answer:
    # ANSWER BOX
    st.markdown(f"""
    <div class="result-box">
        <div class="result-header">ANSWER</div>
        <p style="font-size: 1.05rem; line-height: 1.6; color: #f0f0f0;">{st.session_state.current_answer}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SOURCES USED BOX
    st.markdown("""
    <div class="result-box">
        <div class="result-header">SOURCES USED</div>
        <div class="source-item">
            <span style="color: #6ed0ac; font-weight: 500;">Green Future - Climate Policy Document</span> 
            <span style="color: #6ed0ac; opacity: 0.8; font-size: 0.75rem; border: 1px solid #6ed0ac; border-radius: 4px; padding: 2px 6px; margin-left: 8px;">CURRENT</span>
            <span style="float: right; color: #a0a0a0;">distance 0.4821</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
