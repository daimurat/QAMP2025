"""
RAG/Embedding Management Component

Uses SQLite vector database with Gemini embeddings.
"""
import streamlit as st
import os
from rag import RAGRetriever
from config.constants import RAG_DB_PATH


def render_rag_section():
    """RAG/Embedding section"""
    st.markdown('<div class="sidebar-title">🧩 RAG data & Embeddings</div>', unsafe_allow_html=True)
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    db_exists = os.path.exists(RAG_DB_PATH)
    
    if st.session_state.retriever:
        st.markdown("✅ Gemini Embedding loaded (SQLite)")
    elif db_exists:
        st.markdown("🗂️ RAG database found on disk, but not loaded. Loading now...")
        _load_retriever()
    else:
        st.markdown(f"⚠️ No RAG database found at `{RAG_DB_PATH}`. Please run the QAMP ingestion pipeline first.")
    
    
    # display token usage
    if st.session_state.last_token_count > 0:
        st.markdown("---")
        st.markdown(f"🧮 **Last response token usage:** `{st.session_state.last_token_count}` tokens")


def _load_retriever():
    """Load the RAG retriever from SQLite database."""
    with st.spinner("Loading RAG retriever..."):
        try:
            st.session_state.retriever = RAGRetriever(db_path=RAG_DB_PATH)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load RAG retriever: {e}")
