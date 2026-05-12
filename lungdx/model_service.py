"""Загрузка модели с кэшированием Streamlit."""

import streamlit as st

from config import WEIGHTS_PATH
from src.inference import load_model


@st.cache_resource
def get_model():
    if not WEIGHTS_PATH.exists():
        return None, None, None, None
    try:
        return load_model(str(WEIGHTS_PATH))
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None, None
