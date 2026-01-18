"""
Chat Interface Component
"""
import streamlit as st
from src.utils.helpers import Response
from src.application import workflows
from config.constants import RESPONSE_MODES


def render_chat_interface():
    initial_msg = _get_initial_msg()
    
    _display_chat_history()
    
    if initial_msg:
        _process_user_input(initial_msg)


def _get_initial_msg():
    # check if API key and vector store are available
    has_api_key = st.session_state.get("saved_api_key") or st.session_state.get("saved_api_key_gai")
    
    if has_api_key:
        return st.chat_input("Type your prompt here...")
    else:
        if not has_api_key:
            st.markdown("""
                <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-top: 1rem;">
                    Please enter an API key to use the app
                </div>
            """, unsafe_allow_html=True)
        return None


def _display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _process_user_input(user_input: str):
    # display and save user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.memory.add_user_message(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # count tokens
    _count_tokens(user_input)
    
    # generate assistant response
    with st.chat_message("assistant"):
        response:Response = None
        
        # check generation mode
        mode = st.session_state.get("mode_is_fast", RESPONSE_MODES["FAST"])
        if mode == RESPONSE_MODES["FAST"]:
            # run fast mode
            result, latency = workflows.run_fast_mode(user_input)
            response = Response(content=result)
        if mode == RESPONSE_MODES["DEEP"]:
            result, latency = workflows.run_deep_thought_mode(user_input)
            response = Response(content=result)

        st.markdown(response.content)
        
        st.session_state.memory.add_ai_message(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        
        # rerun if code block is present in response
        if "```" in response.content:
            st.rerun()


def _count_tokens(text: str):
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        st.session_state.last_token_count = len(enc.encode(text))
    except:
        st.session_state.last_token_count = 0
