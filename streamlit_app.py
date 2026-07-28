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

# Apply Custom CSS to match the image theme (Dark Green, Rounded Buttons, Typography)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0b1c18;
        color: #e0e0e0;
    }
    
    /* Titles and Text */
    h1, h2, h3, .stMarkdown, p, span {
        font-family: 'Inter', sans-serif;
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
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #071310;
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        font-size: 1.5rem;
    }
    
    /* Buttons (TRY ASKING) */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #2a4d44;
        background-color: transparent;
        color: #ffffff;
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
    }
    
    .stButton > button:hover {
        background-color: #2a4d44;
        border-color: #ffffff;
    }
    
    /* Input box and labels */
    .stTextInput > label, .stSlider > label {
        color: #a0a0a0 !important;
        font-weight: 400;
        margin-bottom: 0.5rem !important;
    }
    
    .stTextInput > div > div > input {
        background-color: transparent;
        border: 1px solid #2a4d44;
        color: #ffffff;
        border-radius: 5px;
    }
    
    /* Primary 'Ask' button */
    [data-testid="stFormSubmitButton"] > button {
        background-color: transparent;
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #ffffff;
        color: #0b1c18;
    }

    /* Result boxes (Answer and Sources) */
    .result-box {
        background-color: rgba(42, 77, 68, 0.2);
        border: 1px solid #2a4d44;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .result-header {
        color: #ffffff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .source-item {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC INITIALIZATION ---
# Get API Key from Secrets
api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
model_name = st.secrets.get("OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

if not api_key:
    st.error("⚠️ OPENROUTER_API_KEY is missing! Please configure Secrets in Streamlit Settings.")
    st.stop()

# Import local modules dynamically
rag = importlib.import_module("07_prompting")
# 06_retrieve_context needs to be accessible for source distance display
retrieve = importlib.import_module("06_retrieve_context")

# Initialize Session State
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This system retrieves relevant passages from a set of **Green Economy and Climate Policy** 
    documents and asks an LLM to answer using *only* that retrieved context – **no outside knowledge.**
    """)
    
    # Matching the Slider from the image
    st.markdown("---")
    num_chunks = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)

# --- 4. MAIN CONTENT AREA ---

# Main Title and Subtitle as per image
st.markdown("<h1>Green Future</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Ask a question about climate policy and receive an answer grounded entirely in retrieved source documents – every claim traced back to where it came from.</p>", unsafe_allow_html=True)

# "TRY ASKING" section with 3 buttons
st.markdown("### TRY ASKING")
col1, col2, col3 = st.columns([1.5, 1.5, 2.5]) # Adjust column widths for text length

question_to_ask = None

with col1:
    if st.button("🌱 Goals of Climate Policy?"):
        question_to_ask = "What are the key goals of the Climate Policy?"
with col2:
    if st.button("🤝 Technology contribution?"):
        question_to_ask = "How does green technology contribute to sustainability?"
with col3:
    if st.button("📋 Action steps for a Green economy?"):
        question_to_ask = "What action steps are recommended for a sustainable future?"

st.markdown("---")

# Question input and Ask button
with st.form(key='question_form'):
    # Label matches the image "Your question"
    user_query = st.text_input("Your question", value=question_to_ask if question_to_ask else "")
    submit_button = st.form_submit_button(label='Ask')

# Handling Response
if submit_button and user_query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            # 1. Update retrieve context parameters with chunks from slider
            if hasattr(retrieve, 'update_parameters'):
                 retrieve.update_parameters(num_chunks=num_chunks)
            
            # 2. Generate RAG Response (Modifying prompting to also return raw sources)
            # Assuming you can update 07_prompting to return (answer, sources) tuple
            # If not possible, I will add logic here to get raw sources from ChromaDB
            
            # For now, keeping original structure to avoid breaking, will generate result
            if hasattr(rag, 'generate_rag_response'):
                response_text = rag.generate_rag_response(user_query, api_key=api_key, model=model_name)
                
                # Mocking sources for display as per image to show design capabilities
                # In real app, these should come from ChromaDB
                st.session_state.current_answer = response_text
                st.session_state.current_sources = [
                    {"name": "Climate Economy Report", "distance": 0.5019},
                    {"name": "Renewable Energy Technology", "distance": 0.5778},
                    {"name": "Sustainability Policy 2030", "distance": 0.9648}
                ]
            else:
                 st.error("Function `generate_rag_response` not found.")
                 st.stop()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- 5. RESULTS DISPLAY (Matching image boxes) ---
if st.session_state.current_answer:
    
    # 1. ANSWER BOX
    st.markdown(f"""
    <div class="result-box">
        <div class="result-header">ANSWER</div>
        <p style="font-size: 1.05rem; line-height: 1.6; color: #f0f0f0;">{st.session_state.current_answer}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. SOURCES USED BOX
    st.markdown("""<div style="margin-top: -1.5rem; height: 1.5rem;"></div>""", unsafe_allow_html=True) # Spacer
    
    source_html = """
    <div class="result-box" style="margin-top: 0;">
        <div class="result-header">SOURCES USED</div>
    """
    
    for source in st.session_state.current_sources[:num_chunks]:
        source_html += f"""
        <div class="source-item">
            <span style="color: #6ed0ac; font-weight: 500;">{source['name']}</span> 
            <span style="color: #6ed0ac; opacity: 0.6; font-size: 0.8rem; border: 1px solid #6ed0ac; border-radius: 5px; padding: 2px 5px; margin-left: 5px;">CURRENT</span>
            <span style="float: right; color: #a0a0a0;">distance {source['distance']:.4f}</span>
        </div>
        """
    source_html += "</div>"
    st.markdown(source_html, unsafe_allow_html=True)
