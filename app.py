#!/usr/bin/env python3
"""Точка входа Streamlit: LungDx Pro — диагностика лёгочных заболеваний."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from lungdx.styles import inject_styles
from lungdx.ui_main import run

st.set_page_config(
    page_title="LungDx Pro — Диагностика лёгочных заболеваний",
    page_icon="🫁",
    layout="wide",
)

inject_styles()
run()
