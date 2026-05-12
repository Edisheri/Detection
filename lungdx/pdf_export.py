"""Генерация PDF-отчётов (PIL, без Streamlit)."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def build_case_pdf(
    payload: dict[str, Any],
    study_id: str,
    xray_img: Image.Image | None = None,
) -> bytes:
    W, H = 1240, 1754
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    f_title = load_font(44)
    f_text = load_font(28)
    f_small = load_font(22)
    f_tiny = load_font(18)

    draw.rectangle([0, 0, W, 100], fill=(15, 32, 39))
    draw.text((60, 28), "LungDx Pro — Клинический отчёт", fill="white", font=f_title)

    y = 125
    draw.text((60, y), f"Дата: {payload.get('timestamp', '')}", fill="#333", font=f_text)
    y += 42
    draw.text((60, y), f"Идентификатор исследования: {study_id}", fill="#333", font=f_text)
    y += 42
    draw.text((60, y), f"Файл: {payload.get('file_name', '-')}", fill="#333", font=f_text)
    y += 42
    draw.text((60, y), f"Приоритет: {payload.get('priority', '-')}", fill="#333", font=f_text)
    y += 60

    draw.rectangle([40, y, W - 40, y + 2], fill="#00acc1")
    y += 14
    draw.text((60, y), "Результат анализа", fill="#00838f", font=f_text)
    y += 46
    diag = payload.get("predicted_class", "-")
    conf = payload.get("confidence_percent", 0)
    risk = payload.get("risk_level", "-")
    draw.text((60, y), f"Диагноз:        {diag}", fill="black", font=f_text)
    y += 40
    draw.text((60, y), f"Уверенность:    {conf}%", fill="black", font=f_text)
    y += 40
    draw.text((60, y), f"Уровень риска:  {risk}", fill="black", font=f_text)
    y += 55

    draw.text((60, y), "Распределение вероятностей по классам:", fill="#555", font=f_small)
    y += 36
    for cls, val in payload.get("probabilities_percent", {}).items():
        bar_w = int((W - 200) * float(val) / 100)
        draw.rectangle([60, y, 60 + bar_w, y + 24], fill="#00acc1")
        draw.text((60 + bar_w + 8, y), f"{cls}: {val}%", fill="black", font=f_small)
        y += 34
    y += 20

    draw.rectangle([40, y, W - 40, y + 2], fill="#e0e0e0")
    y += 14
    draw.text((60, y), "Контроль качества снимка:", fill="#555", font=f_small)
    y += 32
    q = payload.get("quality", {})
    draw.text(
        (80, y),
        f"Яркость: {q.get('brightness', 0):.1f}   "
        f"Контраст: {q.get('contrast', 0):.1f}   "
        f"Резкость: {q.get('sharpness', 0):.1f}",
        fill="black",
        font=f_small,
    )
    y += 30
    warn = ", ".join(q.get("warnings", [])) if q.get("warnings") else "Замечаний нет"
    draw.text((80, y), f"Замечания: {warn}", fill="black", font=f_small)
    y += 50

    if xray_img is not None:
        thumb = xray_img.resize((220, 220))
        page.paste(thumb, (W - 280, 130))

    draw.rectangle([0, H - 70, W, H], fill=(15, 32, 39))
    draw.text(
        (60, H - 50),
        "Отчёт сформирован системой LungDx Pro. Результат носит справочный характер.",
        fill="white",
        font=f_tiny,
    )

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=100.0)
    return buf.getvalue()


def build_session_pdf(history: list[dict], audit: list[dict]) -> bytes:
    W, H = 1240, 1754
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    f_title = load_font(40)
    f_text = load_font(24)
    f_small = load_font(19)
    f_tiny = load_font(16)

    draw.rectangle([0, 0, W, 90], fill=(15, 32, 39))
    draw.text((60, 22), "LungDx Pro — Сводный отчёт смены", fill="white", font=f_title)

    y = 115
    draw.text((60, y), f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", fill="#333", font=f_text)
    y += 40
    draw.text(
        (60, y),
        f"Исследований: {len(history)}  |  Аудит-записей: {len(audit)}",
        fill="#333",
        font=f_text,
    )
    y += 55

    high = sum(1 for x in history if x.get("Риск") in ("Критический", "Высокий"))
    auto = sum(1 for x in history if x.get("Решение") == "Принять автоматически")
    draw.text(
        (60, y),
        f"Высокий риск: {high}  |  Авто-решения: {auto}",
        fill="#333",
        font=f_text,
    )
    y += 55

    draw.rectangle([40, y, W - 40, y + 2], fill="#00acc1")
    y += 18
    draw.text((60, y), "Журнал исследований смены:", fill="#00838f", font=f_text)
    y += 38
    headers = ["Время", "StudyID", "Диагноз", "Уверен., %", "Риск", "Решение"]
    col_x = [60, 170, 330, 570, 740, 920]
    for hdr, cx in zip(headers, col_x):
        draw.text((cx, y), hdr, fill="#555", font=f_small)
    y += 28
    draw.rectangle([40, y, W - 40, y + 1], fill="#ccc")
    y += 8
    for row in history[-30:]:
        vals = [
            row.get("Время", ""),
            row.get("StudyID", ""),
            row.get("Диагноз", ""),
            str(row.get("Уверенность, %", "")),
            row.get("Риск", ""),
            row.get("Решение", ""),
        ]
        for val, cx in zip(vals, col_x):
            draw.text((cx, y), str(val)[:18], fill="black", font=f_tiny)
        y += 24
        if y > H - 120:
            break

    draw.rectangle([0, H - 70, W, H], fill=(15, 32, 39))
    draw.text(
        (60, H - 50),
        "Отчёт сформирован системой LungDx Pro. Результат носит справочный характер.",
        fill="white",
        font=f_tiny,
    )

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=100.0)
    return buf.getvalue()
