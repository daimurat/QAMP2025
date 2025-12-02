"""
Chat Interface Component
"""
import streamlit as st
from utils.helpers import Response, StreamHandler
from tools.retrieval_tool import RetrievalTool
from workflows.fast import fast_mode_stream
from config.constants import RESPONSE_MODES


def render_chat_interface():
    initial_msg = _get_initial_msg()
    
    _display_chat_history()
    
    if initial_msg:
        _process_user_input(initial_msg)


def _get_initial_msg():
    # check if API key and vector store are available
    has_api_key = st.session_state.get("saved_api_key") or st.session_state.get("saved_api_key_gai")
    has_vector_store = st.session_state.get("vector_store") is not None
    
    if has_api_key and has_vector_store:
        return st.chat_input("Type your prompt here...")
    else:
        if not has_api_key:
            st.markdown("""
                <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-top: 1rem;">
                    Please enter an API key to use the app
                </div>
            """, unsafe_allow_html=True)
        elif not has_vector_store:
            st.markdown("""
                <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-top: 1rem;">
                    Please generate an embedding before using the app
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
    
    # retrieve context
    retrieval_tool = RetrievalTool(vector_store=st.session_state.vector_store)
    context = retrieval_tool.retrieve(user_input)
    
    # count tokens
    _count_tokens(user_input)
    
    # generate assistant response
    with st.chat_message("assistant"):
        response = _generate_response(user_input, context)
        
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


def _generate_response(user_input: str, context: str) -> Response:
    mode = st.session_state.get("mode_is_fast", RESPONSE_MODES["FAST"])
    
    response = None
    if mode == RESPONSE_MODES["FAST"]:
        response = _generate_fast_mode_response(user_input, context)
    if mode == RESPONSE_MODES["DEEP"]:
        # TODO: Implement Deep Thought Mode
        st.warning("Deep Thought Mode is not yet implemented. Using Fast Mode instead.")
    return response


def _generate_fast_mode_response(user_input: str, context: str) -> Response:
    stream_box = st.empty()
    stream_handler = StreamHandler(stream_box)
    
    content = []
    llm_stream = fast_mode_stream(
        user_input, 
        context, 
        st.session_state.memory.messages, 
        stream_handler
    )
    
    for chunk in llm_stream:
        content.append(chunk.content)
    
    full_content = "".join(content)
    response = Response(content=full_content)
    st.markdown(response.content)
    
    return response
