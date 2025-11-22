"""
RAG/Embedding Management Component
"""
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.constants import INDEX_PATH, INDEX_NAME


def render_rag_section():
    """RAG/Embedding section"""
    st.markdown('<div class="sidebar-title">🧩 RAG data & Embeddings</div>', unsafe_allow_html=True)
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    index_exists = os.path.exists(os.path.join(INDEX_PATH, INDEX_NAME + ".faiss"))
    
    if st.session_state.vector_store:
        st.markdown("✅ OpenAI Embedding loaded from file")
    elif index_exists:
        st.markdown("🗂️ OpenAI Embedding file found on disk, but not loaded. Start loading the embedding to use the agents!")
        if not st.session_state.get("saved_api_key"):
            st.error("OpenAI API key is required for loading embeddings. Please enter your OpenAI API key.")
        else:
            _load_embeddings()
    else:
        st.markdown("⚠️ No OpenAI embedding found. Please create the embedding to use the agents!")
    
    
    # display token usage
    if st.session_state.last_token_count > 0:
        st.markdown("---")
        st.markdown(f"🧮 **Last response token usage:** `{st.session_state.last_token_count}` tokens")


def _load_embeddings():
    with st.spinner("Loading embeddings..."):
        embeddings = OpenAIEmbeddings()
        st.session_state.vector_store = FAISS.load_local(
            folder_path=INDEX_PATH,
            index_name=INDEX_NAME,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        st.rerun()
