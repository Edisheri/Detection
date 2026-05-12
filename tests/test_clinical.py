"""Тесты клинических правил (без Streamlit)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lungdx.clinical import (
    decision_status,
    ops_metrics_from_records,
    priority_from_filename,
    priority_rank,
    quality_assessment,
    recommendations,
    risk_level,
    sla_minutes,
)


def test_risk_level_critical_class_high_confidence():
    assert risk_level(0.91, "Cancer") == "Критический"


def test_risk_level_critical_class_lower_confidence():
    assert risk_level(0.85, "Tuberculosis") == "Высокий"


def test_priority_from_filename_stat():
    assert priority_from_filename("STAT_chest.jpg", "Планово") == "STAT (критический)"


def test_priority_rank_ordering():
    assert priority_rank("STAT (критический)") < priority_rank("Планово")


def test_decision_status():
    assert decision_status(0.8, 0.7) == "Принять автоматически"
    assert decision_status(0.5, 0.7) == "Передать на ручную проверку"


def test_recommendations_has_entries():
    assert len(recommendations("Normal")) >= 1


def test_ops_metrics_from_records_empty():
    assert ops_metrics_from_records([]) == {
        "processed": 0,
        "high_risk": 0,
        "auto_rate": 0.0,
        "manual": 0,
    }


def test_ops_metrics_from_records_counts():
    rows = [
        {"Риск": "Высокий", "Решение": "Принять автоматически"},
        {"Риск": "Умеренный", "Решение": "Передать на ручную проверку"},
    ]
    m = ops_metrics_from_records(rows)
    assert m["processed"] == 2
    assert m["high_risk"] == 1
    assert m["auto_rate"] == 50.0
    assert m["manual"] == 1


def test_quality_assessment_runs():
    arr = np.full((64, 64), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    q = quality_assessment(img)
    assert "brightness" in q and "warnings" in q
    assert isinstance(q["warnings"], list)


def test_sla_minutes():
    assert sla_minutes("STAT (критический)") == 10
    assert sla_minutes("Планово") == 24 * 60
