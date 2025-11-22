"""
UI Styling Utilities
"""
import streamlit as st


def inject_global_styles_and_font(font_name: str):
    """Inject global styles and font
    
    Args:
        font_name: Font name to use
    """
    font_url_name = font_name.replace(" ", "+")
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css?family={font_url_name}" rel="stylesheet">
        <style>
        .main-title {{
            font-family: '{font_name}', sans-serif !important;
            font-size: 3.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
            color: var(--text-color);
            letter-spacing: 1px;
        }}
        .sidebar-title {{
            font-family: '{font_name}', sans-serif !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
            color: var(--text-color);
            letter-spacing: 0.5px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    """Render application header"""
    st.markdown(
        '<div class="main-title">qiskit Agent for Pair Programming</div>', 
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/CLAPP.png", width=400)

# Made with Bob
