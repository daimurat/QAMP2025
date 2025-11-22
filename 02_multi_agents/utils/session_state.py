"""
Session State Management for CLAPP
"""
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory


class SessionStateManager:
    """Centralized session state management class"""
    
    # Session state key definitions and default values
    KEYS = {
        "messages": [],
        "debug": False,
        "llm": None,
        "memory": None,  # Set to ChatMessageHistory() during initialization
        "vector_store": None,
        "last_token_count": 0,
        "selected_model": None,
        "greeted": False,
        "debug_messages": [],
        "saved_api_key": None,
        "saved_api_key_gai": None,
        "agents": None,
        "llm_initialized": False,
        "previous_model": None,
        "mode_is_fast": "Fast Mode"
    }
    
    @classmethod
    def initialize(cls):
        """Initialize all session states"""
        for key, default_value in cls.KEYS.items():
            if key not in st.session_state:
                if key == "memory":
                    st.session_state[key] = ChatMessageHistory()
                else:
                    st.session_state[key] = default_value
    
    @staticmethod
    def get(key: str, default=None):
        """Get session state value
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Session state value
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value):
        """Set session state value
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
    
    @staticmethod
    def reset_chat():
        """Reset chat-related states"""
        st.session_state.greeted = False
        st.session_state.messages = []
        st.session_state.memory = ChatMessageHistory()
    
    @staticmethod
    def clear_api_keys():
        """Clear API key-related states"""
        for key in ["saved_api_key", "saved_api_key_gai", "encrypted_key", "encrypted_key_gai"]:
            if key in st.session_state:
                del st.session_state[key]

# Made with Bob
