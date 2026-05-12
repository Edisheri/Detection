"""Тесты генерации PDF (без Streamlit)."""

from __future__ import annotations

from datetime import datetime

from PIL import Image

from lungdx.pdf_export import build_case_pdf, build_session_pdf


def test_build_case_pdf_starts_with_pdf_magic():
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file_name": "test.png",
        "predicted_class": "Норма",
        "confidence_percent": 95.0,
        "risk_level": "Умеренный",
        "priority": "Планово",
        "quality": {"brightness": 128.0, "contrast": 40.0, "sharpness": 200.0, "warnings": []},
        "probabilities_percent": {"Норма": 95.0, "Пневмония": 5.0},
    }
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    data = build_case_pdf(payload, "STUDY123", xray_img=img)
    assert data[:4] == b"%PDF"


def test_build_session_pdf_starts_with_pdf_magic():
    data = build_session_pdf(
        [
            {
                "Время": "10:00:00",
                "StudyID": "ABC",
                "Диагноз": "Норма",
                "Уверенность, %": 90,
                "Риск": "Умеренный",
                "Решение": "Принять автоматически",
            }
        ],
        [],
    )
    assert data[:4] == b"%PDF"
