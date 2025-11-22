"""
Model Selection Component
"""
import streamlit as st
from utils.session_state import SessionStateManager
from config.constants import GPT_MODELS, GEMINI_MODELS, RESPONSE_MODES


def render_model_selector(api_key: str, api_key_gai: str):
    st.markdown('<div class="sidebar-title">Model Selection</div>', unsafe_allow_html=True)
    # list available models
    OPTIONS = []
    if api_key_gai:
        OPTIONS += GEMINI_MODELS
    if api_key:
        OPTIONS += GPT_MODELS
    
    # model selection    
    st.session_state.selected_model = st.selectbox(
        "4. Choose LLM model",
        options=OPTIONS,
        index=0
    )
    
    # model change detection and chat reset
    _handle_model_change()
    
    # LLM initialization flag setting
    _set_llm_initialized(api_key, api_key_gai)
    
    # response mode selection
    _render_response_mode()


def _handle_model_change():
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = st.session_state.selected_model
    elif st.session_state.previous_model != st.session_state.selected_model:
        # reset chat
        SessionStateManager.reset_chat()
        st.session_state.previous_model = st.session_state.selected_model
        st.info("Model changed! Chat has been reset.")


def _set_llm_initialized(api_key: str, api_key_gai: str):
    if st.session_state.selected_model in GEMINI_MODELS:
        if api_key_gai:
            st.session_state.llm_initialized = True
    elif st.session_state.selected_model in GPT_MODELS and api_key:
        st.session_state.llm_initialized = True


def _render_response_mode():
    st.radio(
        label="Response Mode",
        options=RESPONSE_MODES.values(),
        index=0,
        horizontal=True,
        key="mode_is_fast"
    )
    
    st.markdown("<div style='height: 0.5em'></div>", unsafe_allow_html=True)
    st.caption("✨ **Fast Mode**: Single agent setup, quick responses with good quality, but prone to initial errors.")
    st.caption("🎯 **Deep Thought Mode**: Multi-agent setup, responses take longer, more refined, more accurate at first attempt.")
