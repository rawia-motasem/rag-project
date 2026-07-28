__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import importlib

# 1. Import local module
rag = importlib.import_module("07_prompting")

st.set_page_config(page_title="Climate Policy RAG Assistant", page_icon="🌿")

st.title("🌿 Climate Policy & Green Future Assistant")
st.write("Ask questions about climate policy and sustainability, or click one of the suggested questions below!")

# 2. Get API Key from Streamlit Secrets or Environment
api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
model_name = st.secrets.get("OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

if not api_key:
    st.error("⚠️ OPENROUTER_API_KEY is missing! Please configure Secrets in Streamlit Settings.")
    st.stop()

# 3. Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Quick Suggested Questions Buttons ---
st.markdown("### 💡 Suggested Questions:")
col1, col2, col3 = st.columns(3)

prompt_to_send = None

with col1:
    if st.button("📌 Key Goals of Climate Policy"):
        prompt_to_send = "What are the key goals of the Climate Policy?"

with col2:
    if st.button("🌱 Green Technology Role"):
        prompt_to_send = "How does green technology contribute to sustainability?"

with col3:
    if st.button("📋 Action Steps for Future"):
        prompt_to_send = "What action steps are recommended for a sustainable future?"

# Input box for custom questions
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    prompt_to_send = user_input

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User input and response handling
if prompt_to_send:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                if hasattr(rag, 'generate_rag_response'):
                    response_text = rag.generate_rag_response(prompt_to_send, api_key=api_key, model=model_name)
                else:
                    response_text = "Function `generate_rag_response` not found in 07_prompting.py"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
