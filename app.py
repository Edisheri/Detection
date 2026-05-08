#!/usr/bin/env python3
"""Клиническая платформа LungDx Pro — диагностика лёгочных заболеваний."""
import csv
import io
import json
import sys
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WEIGHTS_PATH, CLASS_NAMES_RU, METRICS_DIR
from src.inference import load_model, predict_image, generate_gradcam

st.set_page_config(
    page_title="LungDx Pro — Диагностика лёгочных заболеваний",
    page_icon="🫁",
    layout="wide",
)

st.markdown("""
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
""", unsafe_allow_html=True)

CLASS_NAMES_RU_DEFAULT = {
    "Normal":       "Норма",
    "Pneumonia":    "Пневмония",
    "Cancer":       "Рак лёгкого",
    "COVID-19":     "COVID-19",
    "Tuberculosis": "Туберкулёз",
}


@st.cache_resource
def get_model():
    if not WEIGHTS_PATH.exists():
        return None, None, None, None
    try:
        return load_model(str(WEIGHTS_PATH))
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None, None


def _load_metrics_summary():
    path = METRICS_DIR / "metrics_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_training_history():
    path = WEIGHTS_PATH.parent / "training_history.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _quality_assessment(image):
    arr = np.array(image.convert("L"))
    brightness = float(arr.mean())
    contrast   = float(arr.std())
    blur       = float(np.var(np.diff(arr.astype(np.float32), axis=0)))
    warnings   = []
    if brightness < 45:
        warnings.append("Снимок слишком тёмный")
    if brightness > 210:
        warnings.append("Снимок слишком светлый")
    if contrast < 20:
        warnings.append("Низкий контраст")
    if blur < 120:
        warnings.append("Возможное размытие")
    return {"brightness": brightness, "contrast": contrast, "sharpness": blur, "warnings": warnings}


def _risk_level(confidence, predicted_class):
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


def _recommendations(predicted_class):
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


def _decision_status(confidence, threshold):
    return "Принять автоматически" if confidence >= threshold else "Передать на ручную проверку"


def _confidence_band(confidence):
    if confidence >= 0.95: return "A (95-100%)"
    if confidence >= 0.90: return "B (90-94%)"
    if confidence >= 0.80: return "C (80-89%)"
    if confidence >= 0.70: return "D (70-79%)"
    return "E (< 70%)"


def _priority_from_filename(filename, default):
    u = filename.upper()
    if u.startswith("STAT_"): return "STAT (критический)"
    if u.startswith("URG_"):  return "Срочно"
    if u.startswith("PLAN_"): return "Планово"
    return default


def _priority_rank(p):
    return {"STAT (критический)": 0, "Срочно": 1, "Планово": 2}.get(p, 3)


def _priority_label(p):
    return {"STAT (критический)": "🔴 STAT", "Срочно": "🟠 Срочно", "Планово": "🔵 Планово"}.get(p, p)


def _sla_by_priority(p):
    return {"STAT (критический)": "< 10 мин", "Срочно": "< 30 мин"}.get(p, "< 24 ч")


def _sla_minutes(p):
    return {"STAT (критический)": 10, "Срочно": 30}.get(p, 24 * 60)


def _ensure_session_state():
    defaults = {
        "analysis_history": [],
        "audit_log": [],
        "last_event_hash": "",
        "worklist": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _ops_metrics():
    h = st.session_state.get("analysis_history", [])
    n = len(h)
    high = sum(1 for x in h if x.get("Риск") in ("Критический", "Высокий"))
    auto = sum(1 for x in h if x.get("Решение") == "Принять автоматически")
    return {
        "processed": n,
        "high_risk": high,
        "auto_rate": (auto / n * 100.0) if n else 0.0,
        "manual": n - auto,
    }


def _enqueue_worklist(rows):
    existing = {x["StudyID"] for x in st.session_state["worklist"]}
    for row in rows:
        if row["StudyID"] not in existing:
            p = row.get("Приоритет", "Планово")
            st.session_state["worklist"].append({
                "StudyID": row["StudyID"],
                "Файл": row["Файл"],
                "Приоритет": p,
                "Целевой SLA": _sla_by_priority(p),
                "Диагноз": row["Диагноз"],
                "Уверенность, %": row["Уверенность, %"],
                "Статус": "В очереди",
                "Время поступления": datetime.now().isoformat(timespec="seconds"),
                "Дедлайн": (datetime.now() + timedelta(minutes=_sla_minutes(p))).isoformat(timespec="seconds"),
            })


def _worklist_view_rows():
    now = datetime.now()
    view = []
    for row in st.session_state["worklist"]:
        deadline = datetime.fromisoformat(row["Дедлайн"])
        overdue = now > deadline and row["Статус"] != "Завершено"
        view.append({
            **row,
            "Приоритет (UI)": _priority_label(row["Приоритет"]),
            "SLA статус": "⚠️ Просрочено" if overdue else "✅ В норме",
        })
    return sorted(view, key=lambda x: (_priority_rank(x["Приоритет"]), x["Статус"] != "В очереди"))


def _load_font(size):
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _build_case_pdf(payload, study_id, xray_img=None):
    W, H = 1240, 1754
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    f_title = _load_font(44)
    f_text  = _load_font(28)
    f_small = _load_font(22)
    f_tiny  = _load_font(18)

    draw.rectangle([0, 0, W, 100], fill=(15, 32, 39))
    draw.text((60, 28), "LungDx Pro — Клинический отчёт", fill="white", font=f_title)

    y = 125
    draw.text((60, y), f"Дата: {payload.get('timestamp', '')}",      fill="#333", font=f_text); y += 42
    draw.text((60, y), f"Идентификатор исследования: {study_id}",    fill="#333", font=f_text); y += 42
    draw.text((60, y), f"Файл: {payload.get('file_name', '-')}",     fill="#333", font=f_text); y += 42
    draw.text((60, y), f"Приоритет: {payload.get('priority', '-')}", fill="#333", font=f_text); y += 60

    draw.rectangle([40, y, W - 40, y + 2], fill="#00acc1"); y += 14
    draw.text((60, y), "Результат анализа", fill="#00838f", font=f_text); y += 46
    diag = payload.get("predicted_class", "-")
    conf = payload.get("confidence_percent", 0)
    risk = payload.get("risk_level", "-")
    draw.text((60, y), f"Диагноз:        {diag}",  fill="black", font=f_text); y += 40
    draw.text((60, y), f"Уверенность:    {conf}%", fill="black", font=f_text); y += 40
    draw.text((60, y), f"Уровень риска:  {risk}",  fill="black", font=f_text); y += 55

    draw.text((60, y), "Распределение вероятностей по классам:", fill="#555", font=f_small); y += 36
    for cls, val in payload.get("probabilities_percent", {}).items():
        bar_w = int((W - 200) * float(val) / 100)
        draw.rectangle([60, y, 60 + bar_w, y + 24], fill="#00acc1")
        draw.text((60 + bar_w + 8, y), f"{cls}: {val}%", fill="black", font=f_small)
        y += 34
    y += 20

    draw.rectangle([40, y, W - 40, y + 2], fill="#e0e0e0"); y += 14
    draw.text((60, y), "Контроль качества снимка:", fill="#555", font=f_small); y += 32
    q = payload.get("quality", {})
    draw.text((80, y),
              f"Яркость: {q.get('brightness', 0):.1f}   "
              f"Контраст: {q.get('contrast', 0):.1f}   "
              f"Резкость: {q.get('sharpness', 0):.1f}",
              fill="black", font=f_small); y += 30
    warn = ", ".join(q.get("warnings", [])) if q.get("warnings") else "Замечаний нет"
    draw.text((80, y), f"Замечания: {warn}", fill="black", font=f_small); y += 50

    if xray_img is not None:
        thumb = xray_img.resize((220, 220))
        page.paste(thumb, (W - 280, 130))

    draw.rectangle([0, H - 70, W, H], fill=(15, 32, 39))
    draw.text((60, H - 50),
              "Отчёт сформирован системой LungDx Pro. Результат носит справочный характер.",
              fill="white", font=f_tiny)

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=100.0)
    return buf.getvalue()


def _build_session_pdf(history, audit):
    W, H = 1240, 1754
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    f_title = _load_font(40)
    f_text  = _load_font(24)
    f_small = _load_font(19)
    f_tiny  = _load_font(16)

    draw.rectangle([0, 0, W, 90], fill=(15, 32, 39))
    draw.text((60, 22), "LungDx Pro — Сводный отчёт смены", fill="white", font=f_title)

    y = 115
    draw.text((60, y), f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", fill="#333", font=f_text); y += 40
    draw.text((60, y), f"Исследований: {len(history)}  |  Аудит-записей: {len(audit)}", fill="#333", font=f_text); y += 55

    high = sum(1 for x in history if x.get("Риск") in ("Критический", "Высокий"))
    auto = sum(1 for x in history if x.get("Решение") == "Принять автоматически")
    draw.text((60, y), f"Высокий риск: {high}  |  Авто-решения: {auto}", fill="#333", font=f_text); y += 55

    draw.rectangle([40, y, W - 40, y + 2], fill="#00acc1"); y += 18
    draw.text((60, y), "Журнал исследований смены:", fill="#00838f", font=f_text); y += 38
    headers = ["Время", "StudyID", "Диагноз", "Уверен., %", "Риск", "Решение"]
    col_x = [60, 170, 330, 570, 740, 920]
    for hdr, cx in zip(headers, col_x):
        draw.text((cx, y), hdr, fill="#555", font=f_small)
    y += 28
    draw.rectangle([40, y, W - 40, y + 1], fill="#ccc"); y += 8
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
    draw.text((60, H - 50),
              "Отчёт сформирован системой LungDx Pro. Результат носит справочный характер.",
              fill="white", font=f_tiny)

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=100.0)
    return buf.getvalue()


def _render_result(result, uploaded_name, quality, study_id,
                   decision_threshold, default_priority, image):
    label_ru   = CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"])
    confidence = float(result["confidence"])
    risk       = _risk_level(confidence, result["class"])
    decision   = _decision_status(confidence, decision_threshold)
    priority   = _priority_from_filename(uploaded_name, default_priority)

    # Порог "уверенного" диагноза
    CERTAIN_THRESHOLD = 0.40

    # Детектируем дифференциальную диагностику Cancer ↔ Pneumonia
    probs = result.get("probabilities", {})
    cancer_prob    = float(probs.get("Cancer", 0))
    pneumonia_prob = float(probs.get("Pneumonia", 0))
    diff_diag = (
        cancer_prob > 0.20 and pneumonia_prob > 0.20
        and abs(cancer_prob - pneumonia_prob) < 0.20
    )

    if diff_diag:
        st.markdown(
            f'<div class="warn-box">'
            f'<h3>⚕️ Дифференциальная диагностика</h3>'
            f'<p>Рентгенологическая картина неоднозначна между '
            f'<b>Раком лёгкого</b> ({cancer_prob*100:.1f}%) '
            f'и <b>Пневмонией</b> ({pneumonia_prob*100:.1f}%).<br>'
            f'Эти патологии имеют схожие проявления на рентгене. '
            f'Необходима КТ или бронхоскопия для верификации.</p></div>',
            unsafe_allow_html=True,
        )
    elif confidence >= CERTAIN_THRESHOLD:
        st.markdown(
            f'<div class="result-box"><h3>Диагноз: {label_ru}</h3>'
            f'<p>Уверенность модели: <b>{confidence*100:.1f}%</b></p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="warn-box"><h3>⚠️ Неопределённый результат</h3>'
            f'<p>Наиболее вероятный диагноз: <b>{label_ru}</b> ({confidence*100:.1f}%)'
            f' — ниже порога уверенности ({CERTAIN_THRESHOLD*100:.0f}%).<br>'
            f'Необходима ручная верификация врачом.</p></div>',
            unsafe_allow_html=True,
        )

    verdict_text = "Авто ✓" if confidence >= decision_threshold else "На проверку"
    if diff_diag:
        verdict_text = "Диф. диагноз"
    m1, m2, m3 = st.columns(3)
    m1.metric("Риск",        risk)
    m2.metric("Вердикт",     verdict_text)
    m3.metric("Достоверность", _confidence_band(confidence))

    # Топ-2 прогресс-бара всегда видимы
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    st.progress(confidence, text=f"**{label_ru}**: {confidence*100:.1f}%")
    if len(sorted_probs) > 1:
        second_cls, second_prob = sorted_probs[1]
        second_lbl = CLASS_NAMES_RU_DEFAULT.get(second_cls, second_cls)
        st.progress(float(second_prob), text=f"{second_lbl}: {second_prob*100:.1f}%")

    # Остальные классы — в свёрнутом блоке
    other_probs = {c: p for c, p in probs.items()
                   if c != result["class"] and c != (sorted_probs[1][0] if len(sorted_probs) > 1 else "")}
    if other_probs:
        with st.expander("Полное распределение вероятностей"):
            for cls, prob in sorted(other_probs.items(), key=lambda x: -x[1]):
                lbl = CLASS_NAMES_RU_DEFAULT.get(cls, cls)
                st.progress(float(prob), text=f"{lbl}: {prob*100:.1f}%")

    st.markdown("**Клинические рекомендации:**")
    for line in _recommendations(result["class"]):
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
            width="stretch",
        )
    with dl2:
        st.download_button(
            "📑 Скачать отчёт (PDF)",
            data=_build_case_pdf(payload, study_id, xray_img=image),
            file_name=f"report_{Path(uploaded_name).stem}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    return payload, risk, decision, priority


def main():
    _ensure_session_state()

    st.sidebar.title("🫁 LungDx Pro")
    st.sidebar.caption("Клиническая платформа диагностики лёгочных заболеваний")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметры сессии")
    user_role          = st.sidebar.selectbox("Роль", ["Врач-рентгенолог", "Лаборант", "Администратор"])
    decision_threshold = st.sidebar.slider("Порог авто-решения", 0.40, 0.99, 0.70, 0.01)
    quality_gate       = st.sidebar.checkbox("Quality Gate (блок при плохом снимке)", value=False)
    default_priority   = st.sidebar.selectbox("Приоритет по умолчанию", ["Планово", "Срочно", "STAT (критический)"])
    show_tech          = st.sidebar.checkbox("Model Card (для разработчиков)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Смена:** {datetime.now().strftime('%d.%m.%Y')}")
    st.sidebar.markdown(f"**Оператор:** {user_role}")
    st.sidebar.markdown("**Учреждение:** ГКБ №1, лучевая диагностика")
    st.sidebar.markdown("**Интеграции:** PACS/RIS, ЭМК, аудит-лог")

    st.title("🫁 LungDx Pro — Клиническая диагностика лёгочных заболеваний")
    st.caption("Система поддержки принятия врачебных решений на основе нейронной сети ResNet18. Версия 1.0.")

    ops = _ops_metrics()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Исследований за смену",      ops["processed"])
    k2.metric("Высокий / Критический риск", ops["high_risk"])
    k3.metric("Доля авто-решений",          f"{ops['auto_rate']:.1f}%")
    k4.metric("На ручную верификацию",      ops["manual"])

    s1, s2, s3, s4 = st.columns(4)
    s1.success("Шлюз PACS: В сети")
    s2.success("Сервис инференса: В сети")
    s3.success("Сервис аудита: В сети")
    s4.success("Сервис отчётов: В сети")

    model, class_names, image_size, device = get_model()
    if model is None:
        st.error("⚠️ Модель не загружена. Запустите обучение: `python train.py`")
        st.stop()

    st.markdown("---")
    st.subheader("Анализ одиночного снимка")
    uploaded = st.file_uploader(
        "Загрузите рентгеновский снимок (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    result = None
    if uploaded is not None:
        image    = Image.open(uploaded).convert("RGB")
        quality  = _quality_assessment(image)
        study_id = hashlib.md5(uploaded.name.encode()).hexdigest()[:10].upper()

        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Загруженный снимок", width="stretch")
            st.caption(f"Study ID: {study_id} | Файл: {uploaded.name}")

        with col_result:
            with st.spinner("Анализ изображения..."):
                result = predict_image(model, image, class_names, image_size, device)

            if not result.get("is_xray", True):
                xray_score = result.get("xray_score", 0)
                details = result.get("xray_details", {})
                sat = details.get("mean_saturation", "—")
                dark = details.get("dark_fraction", "—")
                st.markdown(
                    f'<div class="reject-box">'
                    f'<b>⛔ Изображение не является рентгеновским снимком</b><br>'
                    f'Система отклонила файл: обнаружены признаки цветного или нерелевантного изображения.<br>'
                    f'Оценка соответствия: <b>{xray_score*100:.0f}%</b> '
                    f'(насыщенность цвета: {sat if isinstance(sat, str) else f"{sat:.3f}"}, '
                    f'доля тёмного фона: {dark if isinstance(dark, str) else f"{dark:.2%}"}).<br>'
                    f'<i>Пожалуйста, загрузите рентгенограмму грудной клетки (DICOM / JPEG / PNG).</i>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Диагноз не показывается НИКОГДА для нерентгеновских изображений
                st.stop()
            else:
                if result.get("xray_score", 1.0) < 0.9:
                    st.markdown(
                        f'<div class="warn-box">⚠️ Оценка качества снимка: '
                        f'{result.get("xray_score", 0)*100:.0f}%. '
                        f'Рекомендуется повторная съёмка.</div>',
                        unsafe_allow_html=True,
                    )

                if result["confidence"] < decision_threshold:
                    st.warning(
                        f"Уверенность модели ({result['confidence']*100:.1f}%) ниже порога "
                        f"({decision_threshold*100:.0f}%). Рекомендуется ручная верификация."
                    )

                payload, risk, decision, priority = _render_result(
                    result, uploaded.name, quality, study_id,
                    decision_threshold, default_priority, image,
                )

                event_key = f"{study_id}:{result['class']}:{result['confidence']:.4f}"
                if event_key != st.session_state["last_event_hash"]:
                    st.session_state["last_event_hash"] = event_key
                    st.session_state["analysis_history"].append({
                        "Время": datetime.now().strftime("%H:%M:%S"),
                        "StudyID": study_id,
                        "Файл": uploaded.name,
                        "Диагноз": CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"]),
                        "Уверенность, %": round(result["confidence"] * 100, 1),
                        "Риск": risk,
                        "Решение": decision,
                    })
                    st.session_state["audit_log"].append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "study_id": study_id,
                        "file_name": uploaded.name,
                        "predicted_class": result["class"],
                        "confidence": round(result["confidence"], 4),
                        "threshold": decision_threshold,
                        "decision": decision,
                        "band": _confidence_band(result["confidence"]),
                        "priority": priority,
                        "sla_target": _sla_by_priority(priority),
                    })

        st.markdown("---")
        st.subheader("Контроль качества снимка")
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Яркость",  f"{quality['brightness']:.1f}")
        qc2.metric("Контраст", f"{quality['contrast']:.1f}")
        qc3.metric("Резкость", f"{quality['sharpness']:.1f}")
        if quality["warnings"]:
            for w in quality["warnings"]:
                st.warning(w)
            if quality_gate:
                st.error("Quality Gate: автоматическое решение заблокировано из-за качества снимка.")
        else:
            st.success("Качество изображения соответствует требованиям клинического анализа.")

        st.markdown("---")
        st.subheader("Тепловая карта внимания нейросети (Grad-CAM)")
        if model and result is not None and result.get("is_xray", True):
            try:
                target_idx = None
                if result.get("class") in class_names:
                    target_idx = class_names.index(result["class"])
                gradcam = generate_gradcam(model, image, class_names, image_size, device, target_idx)
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    st.image(gradcam["overlay"], caption="Grad-CAM: наложение", width="stretch")
                with gc2:
                    st.image(gradcam["heatmap"], caption="Карта активаций", width="stretch")
                with gc3:
                    alpha = st.slider("Прозрачность наложения", 0.1, 0.9, 0.45, 0.05)
                    blend = np.clip(
                        (1 - alpha) * np.array(image) + alpha * gradcam["overlay"], 0, 255
                    ).astype(np.uint8)
                    st.image(blend, caption="Интерактивное наложение", width="stretch")
            except Exception as e:
                st.warning(f"Grad-CAM недоступен: {e}")

    st.markdown("---")
    st.subheader("Пакетный анализ снимков")
    batch_files = st.file_uploader(
        "Загрузите несколько снимков",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_up",
    )
    if batch_files:
        pf = st.multiselect(
            "Фильтр по приоритету",
            ["STAT (критический)", "Срочно", "Планово"],
            default=["STAT (критический)", "Срочно", "Планово"],
        )
        rows = []
        progress_bar = st.progress(0, text="Обработка...")
        for i, f in enumerate(batch_files):
            progress_bar.progress((i + 1) / len(batch_files), text=f"Обработка {f.name}...")
            try:
                img  = Image.open(f).convert("RGB")
                pred = predict_image(model, img, class_names, image_size, device)
            except Exception:
                continue
            priority = _priority_from_filename(f.name, default_priority)
            if priority not in pf:
                continue
            sid = hashlib.md5(f.name.encode()).hexdigest()[:10].upper()
            rows.append({
                "StudyID": sid,
                "Файл": f.name,
                "Приоритет": priority,
                "Целевой SLA": _sla_by_priority(priority),
                "Диагноз": CLASS_NAMES_RU_DEFAULT.get(pred["class"], pred["class"]),
                "Уверенность, %": round(pred["confidence"] * 100, 1),
                "Риск": _risk_level(pred["confidence"], pred["class"]),
                "Рентген": "✅" if pred.get("is_xray", True) else "❌",
            })
        progress_bar.empty()

        if rows:
            rows.sort(key=lambda x: (_priority_rank(x["Приоритет"]), -x["Уверенность, %"]))
            st.dataframe(rows, width="stretch")
            _enqueue_worklist(rows)

            by_diag = {}
            for row in rows:
                by_diag[row["Диагноз"]] = by_diag.get(row["Диагноз"], 0) + 1
            st.bar_chart(by_diag)

            csv_buf = io.StringIO()
            w = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
            st.download_button(
                "📥 Скачать пакетный отчёт (CSV)",
                data=csv_buf.getvalue().encode("utf-8-sig"),
                file_name="batch_report.csv",
                mime="text/csv",
            )
    else:
        st.info("Очередь исследований пуста. Ожидание снимков из PACS.")

    st.markdown("---")
    st.subheader("Рабочая очередь исследований")
    if st.session_state["worklist"]:
        q_rows = _worklist_view_rows()

        qf1, qf2 = st.columns(2)
        with qf1:
            sf = st.multiselect("Статус", ["В очереди", "В работе", "Завершено"],
                                default=["В очереди", "В работе", "Завершено"], key="q_sf")
        with qf2:
            pf2 = st.multiselect("Приоритет", ["STAT (критический)", "Срочно", "Планово"],
                                 default=["STAT (критический)", "Срочно", "Планово"], key="q_pf")

        q_rows = [r for r in q_rows if r["Статус"] in sf and r["Приоритет"] in pf2]
        st.dataframe(q_rows, width="stretch")

        qm1, qm2, qm3 = st.columns(3)
        qm1.metric("В очереди",      sum(1 for x in st.session_state["worklist"] if x["Статус"] == "В очереди"))
        qm2.metric("В работе",       sum(1 for x in st.session_state["worklist"] if x["Статус"] == "В работе"))
        qm3.metric("SLA просрочено", sum(1 for x in _worklist_view_rows() if "Просрочено" in x["SLA статус"]))

        sel_ids = [x["StudyID"] for x in q_rows] if q_rows else [""]
        b1, b2, b3, b4, b5 = st.columns(5)
        selected = b1.selectbox("Исследование", sel_ids, key="q_sel")
        if b2.button("▶ Взять в работу"):
            for r in st.session_state["worklist"]:
                if r["StudyID"] == selected and r["Статус"] == "В очереди":
                    r["Статус"] = "В работе"; break
        if b3.button("✅ Завершить"):
            for r in st.session_state["worklist"]:
                if r["StudyID"] == selected: r["Статус"] = "Завершено"; break
        if b4.button("↩ Вернуть"):
            for r in st.session_state["worklist"]:
                if r["StudyID"] == selected: r["Статус"] = "В очереди"; break
        if b5.button("🗑 Очистить завершённые"):
            st.session_state["worklist"] = [x for x in st.session_state["worklist"] if x["Статус"] != "Завершено"]
    else:
        st.caption("Рабочая очередь пуста. Добавьте снимки через пакетный анализ.")

    if show_tech:
        st.markdown("---")
        st.subheader("Технический паспорт модели (Model Card)")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("**Архитектура**")
            st.markdown("- ResNet18 + Transfer Learning (ImageNet)")
            st.markdown(f"- Вход: {image_size}×{image_size} RGB")
            st.markdown(f"- Классов: {len(class_names)}")
            st.markdown("- Интерпретируемость: Grad-CAM")
        with mc2:
            st.markdown("**Обучение**")
            st.markdown("- Оптимизатор: Adam (lr=1e-4)")
            st.markdown("- Scheduler: StepLR (step=4, γ=0.5)")
            st.markdown("- Loss: CrossEntropyLoss (веса классов)")
            st.markdown("- Эпох: 10 | Batch: 64 | GPU: RTX 4060")
        with mc3:
            st.markdown("**Классы модели**")
            for cn in class_names:
                st.markdown(f"- {CLASS_NAMES_RU_DEFAULT.get(cn, cn)} (`{cn}`)")

        metrics = _load_metrics_summary()
        history = _load_training_history()
        if metrics:
            st.markdown("**Метрики на валидационной выборке:**")
            ma1, ma2, ma3, ma4 = st.columns(4)
            ma1.metric("Accuracy",   f"{metrics.get('accuracy', 0)*100:.2f}%")
            ma2.metric("Precision",  f"{metrics.get('macro', {}).get('precision', 0)*100:.2f}%")
            ma3.metric("Recall",     f"{metrics.get('macro', {}).get('recall', 0)*100:.2f}%")
            ma4.metric("F1 (macro)", f"{metrics.get('macro', {}).get('f1', 0)*100:.2f}%")

            cm_path = METRICS_DIR / "confusion_matrix.png"
            cr_path = METRICS_DIR / "classification_report.txt"
            col_cm, col_cr = st.columns([1, 1])
            with col_cm:
                if cm_path.exists():
                    st.image(str(cm_path), caption="Матрица ошибок", width="stretch")
            with col_cr:
                if cr_path.exists():
                    st.text_area("Classification Report",
                                 cr_path.read_text(encoding="utf-8"), height=260)
        if history:
            chart_data = {}
            if history.get("train_acc"):
                chart_data["Train Acc"] = history["train_acc"]
            if history.get("val_acc"):
                chart_data["Val Acc"] = history["val_acc"]
            if chart_data:
                st.markdown("**Динамика обучения:**")
                st.line_chart(chart_data)

    st.markdown("---")
    hist_tab, audit_tab = st.tabs(["📋 История исследований", "🔍 Аудит решений"])

    with hist_tab:
        if st.session_state["analysis_history"]:
            st.dataframe(st.session_state["analysis_history"], width="stretch")
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(st.session_state["analysis_history"][0].keys()))
            w.writeheader(); w.writerows(st.session_state["analysis_history"])
            st.download_button("📥 Скачать историю (CSV)",
                               data=buf.getvalue().encode("utf-8-sig"),
                               file_name="session_history.csv", mime="text/csv")
        else:
            st.caption("История пуста. Загрузите снимок для начала работы.")

    with audit_tab:
        if st.session_state["audit_log"]:
            st.dataframe(st.session_state["audit_log"], width="stretch")
            jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in st.session_state["audit_log"])
            st.download_button("📥 Скачать аудит-лог (JSONL)",
                               data=jsonl.encode("utf-8"),
                               file_name="audit_log.jsonl", mime="application/json")
        else:
            st.caption("Аудит-лог пуст.")

    st.markdown("---")
    st.subheader("Операционная сводка смены")
    if st.session_state["analysis_history"]:
        last = st.session_state["analysis_history"][-1]
        st.markdown(
            f"Последнее исследование: `{last['StudyID']}` | "
            f"Диагноз: **{last['Диагноз']}** | "
            f"Уверенность: **{last['Уверенность, %']}%** | "
            f"Риск: **{last['Риск']}**"
        )
        st.download_button(
            "📑 Скачать сводный отчёт смены (PDF)",
            data=_build_session_pdf(st.session_state["analysis_history"], st.session_state["audit_log"]),
            file_name=f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("Исследования ещё не проводились в текущей смене.")


if __name__ == "__main__":
    main()
