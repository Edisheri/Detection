"""Инициализация session_state и операционные метрики."""

from __future__ import annotations

import streamlit as st

from lungdx.clinical import ops_metrics_from_records


def ensure_session_state() -> None:
    defaults = {
        "analysis_history": [],
        "audit_log": [],
        "last_event_hash": "",
        "worklist": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def ops_metrics() -> dict[str, float | int]:
    h = st.session_state.get("analysis_history", [])
    return ops_metrics_from_records(h)
