"""Клинические правила и оценка качества изображения (без зависимости от Streamlit)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def quality_assessment(image: Image.Image) -> dict[str, Any]:
    """Яркость, контраст, «резкость» и текстовые предупреждения для QC-блока."""
    arr = np.array(image.convert("L"))
    brightness = float(arr.mean())
    contrast = float(arr.std())
    blur = float(np.var(np.diff(arr.astype(np.float32), axis=0)))
    warnings: list[str] = []
    if brightness < 45:
        warnings.append("Снимок слишком тёмный")
    if brightness > 210:
        warnings.append("Снимок слишком светлый")
    if contrast < 20:
        warnings.append("Низкий контраст")
    if blur < 120:
        warnings.append("Возможное размытие")
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": blur,
        "warnings": warnings,
    }


def risk_level(confidence: float, predicted_class: str) -> str:
    critical = {"Cancer", "Tuberculosis"}
    if predicted_class in critical:
        if confidence >= 0.90:
            return "Критический"
        return "Высокий"
    if confidence >= 0.95:
        return "Высокий"
    if confidence >= 0.85:
        return "Средний"
    return "Умеренный"


def recommendations(predicted_class: str) -> list[str]:
    recs = {
        "Cancer": [
            "Срочная консультация онколога и торакального хирурга.",
            "Дообследование: КТ грудной клетки с контрастом, онкомаркёры.",
            "Биопсия при наличии показаний.",
        ],
        "Tuberculosis": [
            "Консультация фтизиатра, постановка на учёт.",
            "Бактериологическое и молекулярно-генетическое исследование мокроты.",
            "Пациент нуждается в изоляции до верификации диагноза.",
        ],
        "Pneumonia": [
            "Консультация терапевта / пульмонолога.",
            "Клинико-лабораторная корреляция: CRP, ОАК, прокальцитонин.",
            "Антибактериальная терапия по результатам посева.",
        ],
        "COVID-19": [
            "Оценка сатурации и клинической симптоматики.",
            "ПЦР/экспресс-тест по эпидемиологическому протоколу.",
            "Изоляция пациента до получения результатов анализов.",
        ],
        "Normal": [
            "Патологических изменений на текущем снимке не выявлено.",
            "При сохранении симптомов — динамическое наблюдение.",
        ],
    }
    return recs.get(predicted_class, ["Требуется клиническая верификация результата."])


def decision_status(confidence: float, threshold: float) -> str:
    return "Принять автоматически" if confidence >= threshold else "Передать на ручную проверку"


def confidence_band(confidence: float) -> str:
    if confidence >= 0.95:
        return "A (95-100%)"
    if confidence >= 0.90:
        return "B (90-94%)"
    if confidence >= 0.80:
        return "C (80-89%)"
    if confidence >= 0.70:
        return "D (70-79%)"
    return "E (< 70%)"


def priority_from_filename(filename: str, default: str) -> str:
    u = filename.upper()
    if u.startswith("STAT_"):
        return "STAT (критический)"
    if u.startswith("URG_"):
        return "Срочно"
    if u.startswith("PLAN_"):
        return "Планово"
    return default


def priority_rank(p: str) -> int:
    return {"STAT (критический)": 0, "Срочно": 1, "Планово": 2}.get(p, 3)


def priority_label(p: str) -> str:
    return {
        "STAT (критический)": "🔴 STAT",
        "Срочно": "🟠 Срочно",
        "Планово": "🔵 Планово",
    }.get(p, p)


def sla_by_priority(p: str) -> str:
    return {"STAT (критический)": "< 10 мин", "Срочно": "< 30 мин"}.get(p, "< 24 ч")


def sla_minutes(p: str) -> int:
    return {"STAT (критический)": 10, "Срочно": 30}.get(p, 24 * 60)


def ops_metrics_from_records(history: list[dict]) -> dict[str, float | int]:
    n = len(history)
    high = sum(1 for x in history if x.get("Риск") in ("Критический", "Высокий"))
    auto = sum(1 for x in history if x.get("Решение") == "Принять автоматически")
    return {
        "processed": n,
        "high_risk": high,
        "auto_rate": (auto / n * 100.0) if n else 0.0,
        "manual": n - auto,
    }
