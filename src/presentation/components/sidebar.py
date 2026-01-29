"""
Sidebar Component - Integrates all sidebar sections
"""
import streamlit as st
from src.presentation.components import api_management, model_selector, rag_section


def render_sidebar():
    """Render sidebar

    Returns:
        tuple: (api_key, api_key_gai, api_key_openrouter) - OpenAI, Gemini, and OpenRouter API keys
    """
    with st.sidebar:
        # API management section
        api_key, api_key_gai, api_key_openrouter = api_management.render_api_management()

        st.markdown("---")

        # model selection section
        model_selector.render_model_selector(api_key, api_key_gai, api_key_openrouter)

        st.markdown("---")

        # RAG section
        rag_section.render_rag_section()

    return api_key, api_key_gai, api_key_openrouter
