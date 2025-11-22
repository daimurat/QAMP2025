"""
Sidebar Component - Integrates all sidebar sections
"""
import streamlit as st
from components.api_management import render_api_management
from components.model_selector import render_model_selector
from components.rag_section import render_rag_section


def render_sidebar():
    """Render sidebar
    
    Returns:
        tuple: (api_key, api_key_gai) - OpenAI API key and Gemini API key
    """
    with st.sidebar:
        # API management section
        api_key, api_key_gai = render_api_management()
        
        st.markdown("---")
        
        # model selection section
        render_model_selector(api_key, api_key_gai)
        
        st.markdown("---")
        
        # RAG section
        render_rag_section()
    
    return api_key, api_key_gai
