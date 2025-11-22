"""
Helper Functions and Classes
"""
import json
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


def read_keys_from_file(file_path: str) -> dict:
    """Read keys from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary data
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def read_prompt_from_file(path: str) -> str:
    """Read prompt from file
    
    Args:
        path: Path to prompt file
        
    Returns:
        Prompt text
    """
    with open(path, 'r') as f:
        return f.read()


class Response:
    """Response wrapper class"""
    
    def __init__(self, content: str):
        """
        Args:
            content: Response content
        """
        self.content = content


class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming responses"""
    
    def __init__(self, container):
        """
        Args:
            container: Streamlit container (e.g., st.empty())
        """
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        """Process when new token is received
        
        Args:
            token: Received token
            **kwargs: Other keyword arguments
        """
        self.text += token
        self.container.markdown(self.text + "▌")

# Made with Bob
