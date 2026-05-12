"""Чтение метрик и истории обучения с диска (без Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import METRICS_DIR, WEIGHTS_PATH


def load_metrics_summary(metrics_dir: Path | None = None) -> dict[str, Any] | None:
    path = (metrics_dir or METRICS_DIR) / "metrics_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_training_history(weights_dir: Path | None = None) -> dict[str, Any] | None:
    path = (weights_dir or WEIGHTS_PATH.parent) / "training_history.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
