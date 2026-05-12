"""Отрисовка блока результата анализа (Streamlit)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from lungdx.clinical import (
    confidence_band,
    decision_status,
    priority_from_filename,
    recommendations,
    risk_level,
)
from lungdx.constants import CLASS_NAMES_RU_DEFAULT
from lungdx.pdf_export import build_case_pdf


def render_result(
    result: dict,
    uploaded_name: str,
    quality: dict,
    study_id: str,
    decision_threshold: float,
    default_priority: str,
    image: Image.Image,
) -> tuple[dict, str, str, str]:
    label_ru = CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"])
    confidence = float(result["confidence"])
    risk = risk_level(confidence, result["class"])
    decision = decision_status(confidence, decision_threshold)
    priority = priority_from_filename(uploaded_name, default_priority)

    CERTAIN_THRESHOLD = 0.40

    probs = result.get("probabilities", {})
    cancer_prob = float(probs.get("Cancer", 0))
    pneumonia_prob = float(probs.get("Pneumonia", 0))
    diff_diag = (
        cancer_prob > 0.20
        and pneumonia_prob > 0.20
        and abs(cancer_prob - pneumonia_prob) < 0.20
    )

    if diff_diag:
        st.markdown(
            f'<div class="warn-box">'
            f"<h3>⚕️ Дифференциальная диагностика</h3>"
            f"<p>Рентгенологическая картина неоднозначна между "
            f"<b>Раком лёгкого</b> ({cancer_prob * 100:.1f}%) "
            f"и <b>Пневмонией</b> ({pneumonia_prob * 100:.1f}%).<br>"
            f"Эти патологии имеют схожие проявления на рентгене. "
            f"Необходима КТ или бронхоскопия для верификации.</p></div>",
            unsafe_allow_html=True,
        )
    elif confidence >= CERTAIN_THRESHOLD:
        st.markdown(
            f'<div class="result-box"><h3>Диагноз: {label_ru}</h3>'
            f"<p>Уверенность модели: <b>{confidence * 100:.1f}%</b></p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="warn-box"><h3>⚠️ Неопределённый результат</h3>'
            f"<p>Наиболее вероятный диагноз: <b>{label_ru}</b> ({confidence * 100:.1f}%)"
            f" — ниже порога уверенности ({CERTAIN_THRESHOLD * 100:.0f}%).<br>"
            f"Необходима ручная верификация врачом.</p></div>",
            unsafe_allow_html=True,
        )

    verdict_text = "Авто ✓" if confidence >= decision_threshold else "На проверку"
    if diff_diag:
        verdict_text = "Диф. диагноз"
    m1, m2, m3 = st.columns(3)
    m1.metric("Риск", risk)
    m2.metric("Вердикт", verdict_text)
    m3.metric("Достоверность", confidence_band(confidence))

    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    st.progress(confidence, text=f"**{label_ru}**: {confidence * 100:.1f}%")
    if len(sorted_probs) > 1:
        second_cls, second_prob = sorted_probs[1]
        second_lbl = CLASS_NAMES_RU_DEFAULT.get(second_cls, second_cls)
        st.progress(float(second_prob), text=f"{second_lbl}: {second_prob * 100:.1f}%")

    other_probs = {
        c: p
        for c, p in probs.items()
        if c != result["class"]
        and c != (sorted_probs[1][0] if len(sorted_probs) > 1 else "")
    }
    if other_probs:
        with st.expander("Полное распределение вероятностей"):
            for cls, prob in sorted(other_probs.items(), key=lambda x: -x[1]):
                lbl = CLASS_NAMES_RU_DEFAULT.get(cls, cls)
                st.progress(float(prob), text=f"{lbl}: {prob * 100:.1f}%")

    st.markdown("**Клинические рекомендации:**")
    for line in recommendations(result["class"]):
        st.markdown(f"- {line}")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file_name": uploaded_name,
        "study_id": study_id,
        "predicted_class": label_ru,
        "confidence_percent": round(confidence * 100, 2),
        "risk_level": risk,
        "priority": priority,
        "decision": decision,
        "quality": quality,
        "probabilities_percent": {
            CLASS_NAMES_RU_DEFAULT.get(c, c): round(float(p) * 100, 2)
            for c, p in result["probabilities"].items()
        },
        "system_note": "Результат сформирован клиническим контуром автоматической интерпретации.",
    }

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📄 Скачать отчёт (JSON)",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"report_{Path(uploaded_name).stem}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📑 Скачать отчёт (PDF)",
            data=build_case_pdf(payload, study_id, xray_img=image),
            file_name=f"report_{Path(uploaded_name).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    return payload, risk, decision, priority
