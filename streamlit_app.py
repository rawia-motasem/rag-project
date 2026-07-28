__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

# 1. Import local module
import importlib
rag = importlib.import_module("07_prompting")

st.set_page_config(page_title="RAG Assistant", page_icon="")

st.title(" RAG Assistant")
st.write("Ask questions based on your custom document store!")

# 2. Get API Key from Streamlit Secrets or Environment
api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
model_name = st.secrets.get("OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

if not api_key:
    st.error("⚠️ OPENROUTER_API_KEY is missing! Please configure Secrets in Streamlit Settings.")
    st.stop()

# 3. Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User input and response handling
if user_query := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Call RAG logic from 07_prompting.py
                response_text = rag.generate_rag_response(
                    query=user_query,
                    api_key=api_key,
                    model=model_name
                )
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
