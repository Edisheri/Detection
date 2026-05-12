"""Глобальные стили Streamlit (CSS)."""

import streamlit as st

CSS_MARKDOWN = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-weight: 700; }
    h2, h3 { font-weight: 600; }
    .result-box {
        background: rgba(0, 172, 193, 0.12);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        border-left: 5px solid #00acc1;
    }
    .result-box h3 { margin: 0 0 0.3rem 0; font-size: 1.3rem; }
    .result-box p  { margin: 0; font-size: 1rem; }
    .warn-box {
        background: rgba(255, 152, 0, 0.12);
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
    }
    .reject-box {
        background: rgba(244, 67, 54, 0.12);
        border-left: 5px solid #f44336;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
    }
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS_MARKDOWN, unsafe_allow_html=True)
