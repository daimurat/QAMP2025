"""
CLAPP: qiskit LLM Agent for Pair Programming
Main Application Entry Point
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

import streamlit as st
from dotenv import load_dotenv

from src.utils.session_state import SessionStateManager
# from utils.styles import inject_global_styles_and_font, render_header
from src.presentation.components import inject_global_styles_and_font, render_header, sidebar, chat_interface, welcome_message


# page configuration
st.set_page_config(
    page_title="Qiskit Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# initialization
load_dotenv()
SessionStateManager.initialize()

# style application
inject_global_styles_and_font("Jersey 10")

# header display
render_header()

# sidebar display
api_key, api_key_gai, api_key_openrouter = sidebar.render_sidebar()

# chat interface display
chat_interface.render_chat_interface()

# welcome message display
welcome_message.render_welcome_message(api_key, api_key_gai)
