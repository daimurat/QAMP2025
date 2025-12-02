"""
Welcome Message Component
"""
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.helpers import StreamHandler, read_prompt_from_file
from config.constants import GEMINI_MODELS, PROMPT_PATHS


def render_welcome_message(api_key: str, api_key_gai: str):
    if not _should_show_welcome():
        return
    
    # display welcome message
    with st.chat_message("assistant"):
        welcome_container = st.empty()
        welcome_stream_handler = StreamHandler(welcome_container)
        
        # LLM initialization
        streaming_llm = _initialize_llm(api_key, api_key_gai, welcome_stream_handler)
        
        # generate message
        greeting = _generate_greeting(streaming_llm)
        
        # update session state
        st.session_state.messages.append({"role": "assistant", "content": greeting.content})
        st.session_state.memory.add_ai_message(greeting.content)
        st.session_state.greeted = True
        
        if st.session_state.selected_model in GEMINI_MODELS:
            st.markdown(greeting.content)


def _should_show_welcome() -> bool:
    return (
        st.session_state.get("llm_initialized", False) and
        st.session_state.get("vector_store") is not None and
        not st.session_state.get("greeted", False)
    )


def _initialize_llm(api_key: str, api_key_gai: str, stream_handler):
    if st.session_state.selected_model in GEMINI_MODELS:
        return ChatGoogleGenerativeAI(
            model=st.session_state.selected_model,
            callbacks=[stream_handler],
            temperature=1.0,
            convert_system_message_to_human=True
        )
    else:
        return ChatOpenAI(
            model=st.session_state.selected_model,
            streaming=True,
            callbacks=[stream_handler],
            temperature=1.0
        )


def _generate_greeting(streaming_llm):
    initial_instructions = read_prompt_from_file(PROMPT_PATHS["QISKIT_INSTRUCTIONS"])
    messages = [
        SystemMessage(content=initial_instructions),
        HumanMessage(content="Please greet the user and briefly explain what you can do as the CLASS code assistant.")
    ]
    
    return streaming_llm.invoke(messages)

