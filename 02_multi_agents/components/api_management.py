"""
API Key Management Component
"""
import streamlit as st
import os
import time
from utils.encryption import save_encrypted_key, load_encrypted_key, create_fernet
from config.constants import GPT_MODELS, GEMINI_MODELS


def render_api_management():
    st.markdown('<div class="sidebar-title">🔐 API & Assistants</div>', unsafe_allow_html=True)
    
    api_key = st.text_input("1. OpenAI API Key", type="password")
    api_key_gai = st.text_input("1. Gemini API Key", type="password")
    
    if api_key:
        st.session_state.saved_api_key = api_key
    if api_key_gai:
        st.session_state.saved_api_key_gai = api_key_gai
    
    username = st.text_input("2. Username (for loading or saving API key)", placeholder="Enter your username")
    user_password = st.text_input("3. Password to encrypt/decrypt API key", type="password")
    
    username_display = username if username else 'anon'
    openai_file = f"{username_display}_encrypted_api_key"
    gemini_file = f"{username_display}_gai_encrypted_api_key"
    openai_file_exists = os.path.exists(openai_file)
    gemini_file_exists = os.path.exists(gemini_file)
    
    openai_loaded = bool(st.session_state.get("saved_api_key"))
    gemini_loaded = bool(st.session_state.get("saved_api_key_gai"))
    
    st.markdown(f"OpenAI Key: {'✅ Ready' if openai_loaded else '❌ No Keys'} | Saved: {'🗄️' if openai_file_exists else '—'}")
    st.markdown(f"Gemini Key: {'✅ Ready' if gemini_loaded else '❌ No Keys'} | Saved: {'🗄️' if gemini_file_exists else '—'}")
    
    _render_save_button(username_display, user_password, openai_loaded, gemini_loaded, 
                        openai_file_exists, gemini_file_exists)
    _render_load_button(username, user_password, username_display, 
                        openai_file_exists, gemini_file_exists)
    _render_clear_button(openai_file_exists, gemini_file_exists, openai_file, gemini_file)
    
    if st.session_state.saved_api_key:
        api_key = st.session_state.saved_api_key
    if st.session_state.saved_api_key_gai:
        api_key_gai = st.session_state.saved_api_key_gai
    
    return api_key, api_key_gai


def _render_save_button(username_display, user_password, openai_loaded, gemini_loaded,
                        openai_file_exists, gemini_file_exists):
    if (openai_loaded or gemini_loaded) and user_password and username_display and \
       (not openai_file_exists or not gemini_file_exists):
        if st.button("💾 Save API Key(s) as encrypted file"):
            fernet = create_fernet(user_password)
            try:
                if openai_loaded and not openai_file_exists:
                    encrypted_key = fernet.encrypt(st.session_state.saved_api_key.encode())
                    if save_encrypted_key(encrypted_key.decode(), username_display):
                        st.success("OpenAI API key encrypted and saved! ✅")
                    else:
                        st.warning("OpenAI API key encrypted but couldn't save to file! ⚠️")
                
                if gemini_loaded and not gemini_file_exists:
                    encrypted_key_gai = fernet.encrypt(st.session_state.saved_api_key_gai.encode())
                    if save_encrypted_key(encrypted_key_gai.decode(), username_display+'_gai'):
                        st.success("Gemini API key encrypted and saved! ✅")
                    else:
                        st.warning("Gemini API key encrypted but couldn't save to file! ⚠️")
            except Exception as e:
                st.error(f"Error saving API key: {str(e)}")
            st.rerun()


def _render_load_button(username, user_password, username_display, 
                        openai_file_exists, gemini_file_exists):
    if st.button("🔐 Load Saved API Key(s)"):
        if not username or not user_password:
            st.error("Please enter both username and password to load saved API key(s).")
        else:
            fernet = create_fernet(user_password)
            error = False
            try:
                if openai_file_exists:
                    encrypted_key = load_encrypted_key(username_display)
                    if encrypted_key:
                        decrypted_key = fernet.decrypt(encrypted_key.encode()).decode()
                        st.session_state.saved_api_key = decrypted_key
                        st.success("OpenAI API key loaded from encrypted file! 🔑")
                
                if gemini_file_exists:
                    encrypted_key_gai = load_encrypted_key(username_display+'_gai')
                    if encrypted_key_gai:
                        decrypted_key_gai = fernet.decrypt(encrypted_key_gai.encode()).decode()
                        st.session_state.saved_api_key_gai = decrypted_key_gai
                        st.success("Gemini API key loaded from encrypted file! 🔑")
            except Exception:
                st.error("Failed to decrypt API key(s): Please check your username and password.")
                error = True
            
            if not error:
                # Set llm_initialized if a key is loaded and a model is selected
                if ((st.session_state.saved_api_key and st.session_state.selected_model in GPT_MODELS) or
                    (st.session_state.saved_api_key_gai and st.session_state.selected_model in GEMINI_MODELS)):
                    st.session_state.llm_initialized = True
                st.rerun()


def _render_clear_button(openai_file_exists, gemini_file_exists, openai_file, gemini_file):
    if openai_file_exists or gemini_file_exists:
        if st.button("🗑️ Clear Saved API Key(s)"):
            deleted_files = False
            error_message = ""
            try:
                if openai_file_exists:
                    os.remove(openai_file)
                    deleted_files = True
                if gemini_file_exists:
                    os.remove(gemini_file)
                    deleted_files = True
            except Exception as e:
                error_message += f"Error clearing file: {str(e)}\n"
            
            # Clear session state
            for k in ["saved_api_key", "saved_api_key_gai", "encrypted_key", "encrypted_key_gai"]:
                if k in st.session_state:
                    del st.session_state[k]
            
            if deleted_files:
                st.info("Saved API key(s) cleared. Reloading page...")
                time.sleep(1)
                st.rerun()
            elif error_message:
                st.error(error_message)
            else:
                st.warning("No saved API keys found to delete.")
