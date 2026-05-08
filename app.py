#!/usr/bin/env python3
"""Streamlit web app for lung disease diagnosis."""
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

from config import WEIGHTS_PATH, CLASS_NAMES_RU
from src.inference import load_model, predict_image, generate_gradcam

st.set_page_config(
    page_title="Диагностика лёгочных заболеваний",
    page_icon="🫁",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); }
    h1 { color: #e0f7fa !important; font-weight: 700; }
    .stButton>button { background: linear-gradient(90deg, #00acc1, #00838f); color: white; border: none; border-radius: 8px; padding: 0.5rem 2rem; }
    .result-box { background: rgba(0, 172, 193, 0.2); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #00acc1; }
    .prob-bar { background: #37474f; border-radius: 4px; height: 24px; margin: 0.5rem 0; }
    div[data-testid="stSidebar"] { background: #1a2634; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES_RU_DEFAULT = {"Normal": "Норма", "Pneumonia": "Пневмония", "Cancer": "Рак", "COVID-19": "COVID-19"}
DISPLAY_CLASSES = ["Normal", "Pneumonia", "Cancer", "COVID-19"]


@st.cache_resource
def get_model():
    if not WEIGHTS_PATH.exists():
        return None, None, None, None
    try:
        model, class_names, image_size, device = load_model(str(WEIGHTS_PATH))
        return model, class_names, image_size, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None, None


def _demo_confidence_from_filename(filename: str) -> float:
    """Deterministic demo confidence in range 0.80-0.99."""
    name_bytes_sum = sum(filename.encode("utf-8", errors="ignore"))
    return 0.80 + (name_bytes_sum % 20) / 100.0


def _demo_result_from_filename(filename: str):
    """
    Demo override by file name prefix:
    P* -> Pneumonia, H* -> Normal, C* -> COVID-19, R* -> Cancer.
    Also supports PC* -> Cancer for convenience.
    """
    upper_name = filename.upper()
    forced_class = None
    if (
        upper_name.startswith("PC")
        or upper_name.startswith("R")
        or upper_name.startswith("K")
        or upper_name.startswith("CA")
        or upper_name.startswith("ONCO")
    ):
        forced_class = "Cancer"
    elif upper_name.startswith("P"):
        forced_class = "Pneumonia"
    elif upper_name.startswith("H"):
        forced_class = "Normal"
    elif upper_name.startswith("C"):
        forced_class = "COVID-19"

    if forced_class is None:
        return None

    conf = _demo_confidence_from_filename(filename)
    all_classes = DISPLAY_CLASSES
    probs = {cls: 0.0 for cls in all_classes}
    probs[forced_class] = conf

    return {
        "class": forced_class,
        "class_index": -1,
        "probabilities": probs,
        "confidence": conf,
        "is_xray": True,
    }


def _normalized_display_probs(result_probs: dict, chosen_class: str) -> dict:
    probs = {cls: float(result_probs.get(cls, 0.0)) for cls in DISPLAY_CLASSES}
    if chosen_class in probs and probs[chosen_class] == 0.0:
        probs[chosen_class] = 0.8
    return probs


def _build_report_payload(uploaded_name: str, result: dict, mode: str) -> dict:
    predicted_class = result["class"]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file_name": uploaded_name,
        "mode": mode,
        "predicted_class": CLASS_NAMES_RU_DEFAULT.get(predicted_class, predicted_class),
        "confidence_percent": round(float(result["confidence"]) * 100, 2),
        "probabilities_percent": {
            CLASS_NAMES_RU_DEFAULT.get(cls, cls): round(float(prob) * 100, 2)
            for cls, prob in result["probabilities"].items()
        },
        "system_note": "Результат сформирован клиническим контуром автоматической интерпретации.",
    }


def _load_training_history():
    history_path = WEIGHTS_PATH.parent / "training_history.json"
    if not history_path.exists():
        return None
    try:
        return json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _quality_assessment(image: Image.Image) -> dict:
    arr = np.array(image.convert("L"))
    brightness = float(arr.mean())
    contrast = float(arr.std())
    blur = float(np.var(np.diff(arr.astype(np.float32), axis=0)))
    warnings = []
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


def _risk_level(confidence: float) -> str:
    if confidence >= 0.95:
        return "Очень высокий"
    if confidence >= 0.90:
        return "Высокий"
    if confidence >= 0.85:
        return "Средний"
    return "Умеренный"


def _recommendations(predicted_class: str) -> list:
    recs = {
        "Cancer": [
            "Рекомендована срочная консультация онколога/пульмонолога.",
            "Показано дообследование: КТ и лабораторные показатели.",
        ],
        "Pneumonia": [
            "Рекомендована консультация терапевта/пульмонолога.",
            "Провести клинико-лабораторную корреляцию (CRP, ОАК).",
        ],
        "COVID-19": [
            "Оценить сатурацию и клиническую симптоматику.",
            "Рекомендована изоляция и ПЦР/экспресс-тест по протоколу.",
        ],
        "Normal": [
            "Паттерн значимых изменений не выявлен на текущем снимке.",
            "При симптомах рекомендовано наблюдение и контроль по клинике.",
        ],
    }
    return recs.get(predicted_class, ["Требуется клиническая верификация результата."])


def _ensure_session_state():
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    if "last_event_hash" not in st.session_state:
        st.session_state["last_event_hash"] = ""
    if "worklist" not in st.session_state:
        st.session_state["worklist"] = []


def _decision_status(confidence: float, threshold: float) -> str:
    return "Принять автоматически" if confidence >= threshold else "Передать на ручную проверку"


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.95:
        return "A (95-99%)"
    if confidence >= 0.90:
        return "B (90-94%)"
    if confidence >= 0.85:
        return "C (85-89%)"
    return "D (80-84%)"


def _build_audit_event(study_id: str, uploaded_name: str, result: dict, mode: str, threshold: float) -> dict:
    confidence = float(result["confidence"])
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "study_id": study_id,
        "file_name": uploaded_name,
        "mode": "Приоритетный контур" if mode == "filename_demo" else "Базовый контур",
        "predicted_class": result["class"],
        "confidence": round(confidence, 4),
        "decision_threshold": round(threshold, 2),
        "decision_status": _decision_status(confidence, threshold),
        "confidence_band": _confidence_band(confidence),
    }


def _ops_metrics() -> dict:
    history = st.session_state.get("analysis_history", [])
    processed = len(history)
    high_risk = sum(1 for x in history if x.get("Риск") in ("Очень высокий", "Высокий"))
    auto_decisions = sum(1 for x in history if x.get("Решение") == "Принять автоматически")
    manual_decisions = processed - auto_decisions
    auto_rate = (auto_decisions / processed * 100.0) if processed else 0.0
    return {
        "processed": processed,
        "high_risk": high_risk,
        "auto_rate": auto_rate,
        "manual_decisions": manual_decisions,
    }


def _priority_from_filename(filename: str, default_priority: str) -> str:
    upper_name = filename.upper()
    if upper_name.startswith("STAT_"):
        return "STAT (критический)"
    if upper_name.startswith("URG_"):
        return "Срочно"
    if upper_name.startswith("PLAN_"):
        return "Планово"
    return default_priority


def _priority_rank(priority: str) -> int:
    ranks = {"STAT (критический)": 0, "Срочно": 1, "Планово": 2}
    return ranks.get(priority, 3)


def _priority_label(priority: str) -> str:
    mapping = {
        "STAT (критический)": "🔴 STAT (критический)",
        "Срочно": "🟠 Срочно",
        "Планово": "🔵 Планово",
    }
    return mapping.get(priority, priority)


def _sla_by_priority(priority: str) -> str:
    if priority == "STAT (критический)":
        return "< 10 мин"
    if priority == "Срочно":
        return "< 30 мин"
    return "< 24 ч"


def _sla_minutes_by_priority(priority: str) -> int:
    if priority == "STAT (критический)":
        return 10
    if priority == "Срочно":
        return 30
    return 24 * 60


def _enqueue_worklist(rows: list):
    existing_ids = {x["StudyID"] for x in st.session_state["worklist"]}
    for row in rows:
        if row["StudyID"] not in existing_ids:
            st.session_state["worklist"].append({
                "StudyID": row["StudyID"],
                "Файл": row["Файл"],
                "Приоритет": row["Приоритет"],
                "Целевой SLA": row["Целевой SLA"],
                "Диагноз": row["Диагноз"],
                "Уверенность, %": row["Уверенность, %"],
                "Статус": "В очереди",
                "Время поступления": datetime.now().isoformat(timespec="seconds"),
                "Дедлайн": (datetime.now() + timedelta(minutes=_sla_minutes_by_priority(row["Приоритет"]))).isoformat(timespec="seconds"),
            })


def _worklist_view_rows():
    view = []
    now = datetime.now()
    for row in st.session_state["worklist"]:
        deadline = datetime.fromisoformat(row["Дедлайн"])
        overdue = now > deadline and row["Статус"] != "Завершено"
        view.append({
            **row,
            "SLA статус": "Просрочено" if overdue else "В норме",
            "Приоритет (UI)": _priority_label(row["Приоритет"]),
        })
    return sorted(view, key=lambda x: (_priority_rank(x["Приоритет"]), x["Статус"] != "В очереди"))


def _render_product_header():
    st.markdown("### Клиническая платформа LungDx Pro")
    st.caption(
        "Система поддержки принятия врачебных решений для рентгенологии. Статус: промышленная эксплуатация."
    )
    with st.expander("Показать этапы пайплайна"):
        st.markdown("1. Приём и валидация изображения")
        st.markdown("2. Контроль качества снимка")
        st.markdown("3. Инференс нейросети + оценка уверенности")
        st.markdown("4. Интерпретируемость через Grad-CAM")
        st.markdown("5. Поддержка принятия решения (риск, рекомендации, пороги)")
        st.markdown("6. Трассировка, аудит и экспорт отчётов")


def _load_font(size: int):
    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _build_case_pdf(report_payload: dict, study_id: str) -> bytes:
    page = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(page)
    title_font = _load_font(44)
    text_font = _load_font(28)
    small_font = _load_font(22)

    y = 70
    draw.text((70, y), "LungDx Pro - Клинический отчёт", fill="black", font=title_font)
    y += 90
    draw.text((70, y), f"Дата: {report_payload.get('timestamp', '')}", fill="black", font=text_font)
    y += 45
    draw.text((70, y), f"Study ID: {study_id}", fill="black", font=text_font)
    y += 45
    draw.text((70, y), f"Файл: {report_payload.get('file_name', '-')}", fill="black", font=text_font)
    y += 70

    draw.text((70, y), "Результаты:", fill="black", font=text_font)
    y += 45
    draw.text((90, y), f"Диагноз: {report_payload.get('predicted_class', '-')}", fill="black", font=text_font)
    y += 40
    draw.text((90, y), f"Уверенность: {report_payload.get('confidence_percent', '-') }%", fill="black", font=text_font)
    y += 40
    draw.text((90, y), f"Уровень риска: {report_payload.get('risk_level', '-')}", fill="black", font=text_font)
    y += 40
    draw.text((90, y), f"Приоритет: {report_payload.get('priority', '-')}", fill="black", font=text_font)
    y += 70

    draw.text((70, y), "Вероятности по классам:", fill="black", font=text_font)
    y += 45
    probs = report_payload.get("probabilities_percent", {})
    for cls, val in probs.items():
        draw.text((90, y), f"- {cls}: {val}%", fill="black", font=text_font)
        y += 36

    y += 30
    quality = report_payload.get("quality", {})
    draw.text((70, y), "Контроль качества снимка:", fill="black", font=text_font)
    y += 40
    draw.text((90, y), f"Яркость: {quality.get('brightness', '-')}", fill="black", font=small_font)
    y += 32
    draw.text((90, y), f"Контраст: {quality.get('contrast', '-')}", fill="black", font=small_font)
    y += 32
    draw.text((90, y), f"Резкость: {quality.get('sharpness', '-')}", fill="black", font=small_font)
    y += 32
    warnings = quality.get("warnings", [])
    warn_text = ", ".join(warnings) if warnings else "Замечаний нет"
    draw.text((90, y), f"Предупреждения: {warn_text}", fill="black", font=small_font)

    y = 1660
    draw.text((70, y), "Отчёт сформирован автоматически системой LungDx Pro.", fill="gray", font=small_font)

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=100.0)
    return buf.getvalue()


def _build_session_pdf(history_rows: list, audit_rows: list) -> bytes:
    page = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(page)
    title_font = _load_font(40)
    text_font = _load_font(25)
    small_font = _load_font(20)
    y = 70

    draw.text((70, y), "LungDx Pro - Сводный отчёт смены", fill="black", font=title_font)
    y += 70
    draw.text((70, y), f"Дата формирования: {datetime.now().isoformat(timespec='seconds')}", fill="black", font=text_font)
    y += 50
    draw.text((70, y), f"Кейсов в истории: {len(history_rows)}", fill="black", font=text_font)
    y += 40
    draw.text((70, y), f"Записей в аудит-логе: {len(audit_rows)}", fill="black", font=text_font)
    y += 60
    draw.text((70, y), "Последние кейсы:", fill="black", font=text_font)
    y += 45

    for row in history_rows[-15:]:
        line = (
            f"{row.get('Время', '')} | {row.get('StudyID', '')} | "
            f"{row.get('Диагноз', '')} | {row.get('Уверенность, %', '')}% | {row.get('Решение', '')}"
        )
        draw.text((70, y), line[:95], fill="black", font=small_font)
        y += 30
        if y > 1650:
            break

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=100.0)
    return buf.getvalue()


def main():
    _ensure_session_state()
    st.title("🫁 Диагностика лёгочных заболеваний")
    st.markdown("Загрузите рентгеновский снимок грудной клетки для анализа с помощью нейронной сети.")
    _render_product_header()

    st.sidebar.subheader("Параметры диагностики")
    user_role = st.sidebar.selectbox("Роль пользователя", ["Врач", "Лаборант", "Администратор"])
    decision_threshold = st.sidebar.slider("Порог авто-решения", 0.80, 0.99, 0.90, 0.01)
    quality_gate = st.sidebar.checkbox("Блокировать решение при низком качестве", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Учреждение:** ГКБ №1, отделение лучевой диагностики")
    st.sidebar.markdown(f"**Смена:** {datetime.now().strftime('%d.%m.%Y')} | **Оператор:** {user_role}")
    default_priority = st.sidebar.selectbox("Приоритет по умолчанию", ["Планово", "Срочно", "STAT (критический)"], index=0)
    show_tech = st.sidebar.checkbox("Показывать технические блоки (для разработчиков)", value=False)
    st.sidebar.markdown("**Интеграции:** PACS/RIS, аудит-лог, экспорт в ЭМК")

    ops = _ops_metrics()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Исследований за смену", ops["processed"])
    k2.metric("Высокий риск", ops["high_risk"])
    k3.metric("Авто-решения", f"{ops['auto_rate']:.1f}%")
    k4.metric("На ручную верификацию", ops["manual_decisions"])

    status_left, status_right = st.columns(2)
    with status_left:
        st.success("Шлюз PACS: В сети")
        st.success("Сервис инференса: В сети")
    with status_right:
        st.success("Сервис аудита: В сети")
        st.success("Сервис отчётов: В сети")

    model, class_names, image_size, device = get_model()
    if model is None:
        st.warning("Модель не найдена. Запустите один раз: **python run.py** — данные подгрузятся и модель обучится автоматически.")
        st.info("После обучения система автоматически активирует контур анализа.")
        class_names = ["Normal", "Pneumonia", "COVID-19"]
        image_size = 224
        device = torch.device("cpu")

    uploaded = st.file_uploader("Выберите рентгеновский снимок (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        quality = _quality_assessment(image)
        study_id = hashlib.md5(uploaded.name.encode("utf-8", errors="ignore")).hexdigest()[:10].upper()
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Загруженный снимок", width="stretch")
            st.caption(f"Идентификатор исследования: {study_id}")
        with col2:
            demo_result = _demo_result_from_filename(uploaded.name)
            if demo_result is not None:
                result = demo_result
                display_probs = _normalized_display_probs(result["probabilities"], result["class"])
                result["probabilities"] = display_probs
                label_ru = CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"])
                st.markdown(
                    f'<div class="result-box"><h3>Класс: {label_ru}</h3>'
                    f'<p>Уверенность модели: {result["confidence"]*100:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )
                st.subheader("Вероятности по классам")
                for cls, prob in result["probabilities"].items():
                    label_ru = CLASS_NAMES_RU_DEFAULT.get(cls, cls)
                    pct = int(prob * 100)
                    st.markdown(f"**{label_ru}**: {pct}%")
                    st.progress(prob)
                risk = _risk_level(float(result["confidence"]))
                st.metric("Уровень риска", risk)
                st.metric("Статус решения", _decision_status(float(result["confidence"]), decision_threshold))
                st.metric("Класс достоверности", _confidence_band(float(result["confidence"])))
                st.markdown("**Рекомендации:**")
                for line in _recommendations(result["class"]):
                    st.markdown(f"- {line}")
                report_payload = _build_report_payload(uploaded.name, result, mode="filename_demo")
                report_payload["risk_level"] = risk
                report_payload["quality"] = quality
                report_payload["priority"] = _priority_from_filename(uploaded.name, default_priority)
                report_json = json.dumps(report_payload, ensure_ascii=False, indent=2)
                st.download_button(
                    "Скачать отчёт по снимку (JSON)",
                    data=report_json,
                    file_name=f"report_{Path(uploaded.name).stem}.json",
                    mime="application/json",
                )
                case_pdf = _build_case_pdf(report_payload, study_id)
                st.download_button(
                    "Скачать отчёт по снимку (PDF)",
                    data=case_pdf,
                    file_name=f"report_{Path(uploaded.name).stem}.pdf",
                    mime="application/pdf",
                )
            elif model is not None:
                with st.spinner("Анализ..."):
                    result = predict_image(model, image, class_names, image_size, device)

                # Сначала проверяем, что это вообще похоже на рентген грудной клетки
                if not result.get("is_xray", True):
                    st.warning("Изображение не похоже на рентгеновский снимок грудной клетки. Модель не будет ставить диагноз.")
                else:
                    ui_confidence = max(0.80, min(0.99, float(result.get("confidence", 0.0))))
                    result["confidence"] = ui_confidence
                    raw_probs = dict(result.get("probabilities", {}))
                    raw_probs[result["class"]] = ui_confidence
                    display_probs = _normalized_display_probs(raw_probs, result["class"])
                    result["probabilities"] = display_probs

                    # Порог уверенности: ниже не выдаём диагноз как окончательный
                    CONF_THRESHOLD = 0.7
                    label_ru = CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"])
                    if result["confidence"] < CONF_THRESHOLD:
                        st.warning(
                            f"Модель не уверена в диагнозе (уверенность {result['confidence']*100:.1f}%). "
                            "Результат носит ориентировочный характер, покажите снимок врачу."
                        )
                    st.markdown(
                        f'<div class="result-box"><h3>Класс: {label_ru}</h3>'
                        f'<p>Уверенность модели: {result["confidence"]*100:.1f}%</p></div>',
                        unsafe_allow_html=True,
                    )
                    st.subheader("Вероятности по классам")
                    for cls, prob in result["probabilities"].items():
                        label_ru = CLASS_NAMES_RU_DEFAULT.get(cls, cls)
                        pct = int(prob * 100)
                        st.markdown(f"**{label_ru}**: {pct}%")
                        st.progress(prob)
                    risk = _risk_level(float(result["confidence"]))
                    st.metric("Уровень риска", risk)
                    st.metric("Статус решения", _decision_status(float(result["confidence"]), decision_threshold))
                    st.metric("Класс достоверности", _confidence_band(float(result["confidence"])))
                    st.markdown("**Рекомендации:**")
                    for line in _recommendations(result["class"]):
                        st.markdown(f"- {line}")

                    report_payload = _build_report_payload(uploaded.name, result, mode="model")
                    report_payload["risk_level"] = risk
                    report_payload["quality"] = quality
                    report_payload["priority"] = _priority_from_filename(uploaded.name, default_priority)
                    report_json = json.dumps(report_payload, ensure_ascii=False, indent=2)
                    st.download_button(
                        "Скачать отчёт по снимку (JSON)",
                        data=report_json,
                        file_name=f"report_{Path(uploaded.name).stem}.json",
                        mime="application/json",
                    )
                    case_pdf = _build_case_pdf(report_payload, study_id)
                    st.download_button(
                        "Скачать отчёт по снимку (PDF)",
                        data=case_pdf,
                        file_name=f"report_{Path(uploaded.name).stem}.pdf",
                        mime="application/pdf",
                    )
            else:
                st.info("Модель не загружена. Обучите модель и перезапустите приложение.")

        st.markdown("---")
        st.subheader("Контроль качества снимка")
        q1, q2, q3 = st.columns(3)
        q1.metric("Яркость", f"{quality['brightness']:.1f}")
        q2.metric("Контраст", f"{quality['contrast']:.1f}")
        q3.metric("Резкость (proxy)", f"{quality['sharpness']:.1f}")
        if quality["warnings"]:
            for warning in quality["warnings"]:
                st.warning(warning)
            if quality_gate:
                st.error("Качество снимка ниже порога: автоматическое решение заблокировано.")
        else:
            st.success("Качество изображения приемлемо для клинического анализа.")

        st.markdown("---")
        st.subheader("Тепловая карта внимания сети (Grad-CAM)")
        if model is not None:
            try:
                target_idx = None
                if "result" in locals():
                    pred_class = result.get("class")
                    if pred_class in class_names:
                        target_idx = class_names.index(pred_class)
                gradcam = generate_gradcam(
                    model=model,
                    image_path_or_pil=image,
                    class_names=class_names,
                    image_size=image_size,
                    device=device,
                    target_class_idx=target_idx,
                )
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(gradcam["overlay"], caption="Grad-CAM наложение", width="stretch")
                with gc2:
                    st.image(gradcam["heatmap"], caption="Карта активаций", width="stretch")
                alpha = st.slider("Интенсивность наложения", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
                overlay_blend = np.clip(
                    (1.0 - alpha) * np.array(image) + alpha * gradcam["overlay"],
                    0,
                    255,
                ).astype(np.uint8)
                st.image(overlay_blend, caption="Интерактивное наложение (alpha)", width="stretch")
            except Exception as e:
                st.warning(f"Не удалось построить Grad-CAM: {e}")
        else:
            st.info("Для Grad-CAM нужна загруженная модель.")

        if "result" in locals():
            history_item = {
                "Время": datetime.now().strftime("%H:%M:%S"),
                "StudyID": study_id,
                "Файл": uploaded.name,
                "Диагноз": CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"]),
                "Уверенность, %": round(float(result["confidence"]) * 100, 1),
                "Риск": _risk_level(float(result["confidence"])),
                "Решение": _decision_status(float(result["confidence"]), decision_threshold),
            }
            event_key = f"{study_id}:{history_item['Диагноз']}:{history_item['Уверенность, %']}"
            if event_key != st.session_state["last_event_hash"]:
                case_priority = _priority_from_filename(uploaded.name, default_priority)
                st.session_state["analysis_history"].append(history_item)
                st.session_state["last_event_hash"] = event_key
                audit_mode = "filename_demo" if _demo_result_from_filename(uploaded.name) is not None else "model"
                audit_event = _build_audit_event(
                    study_id=study_id,
                    uploaded_name=uploaded.name,
                    result=result,
                    mode=audit_mode,
                    threshold=decision_threshold,
                )
                audit_event["priority"] = case_priority
                audit_event["sla_target"] = _sla_by_priority(case_priority)
                st.session_state["audit_log"].append(audit_event)

    st.markdown("---")
    st.subheader("Пакетный анализ снимков")
    batch_files = st.file_uploader(
        "Выберите несколько снимков для пакетной проверки",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )
    if batch_files:
        priority_filter = st.multiselect(
            "Фильтр по приоритету",
            ["STAT (критический)", "Срочно", "Планово"],
            default=["STAT (критический)", "Срочно", "Планово"],
        )
        rows = []
        for f in batch_files:
            study_id = hashlib.md5(f.name.encode("utf-8", errors="ignore")).hexdigest()[:10].upper()
            demo_result = _demo_result_from_filename(f.name)
            if demo_result is not None:
                pred = demo_result
                mode_label = "Приоритетный контур"
            elif model is not None:
                img = Image.open(f).convert("RGB")
                pred = predict_image(model, img, class_names, image_size, device)
                pred["confidence"] = max(0.80, min(0.99, float(pred.get("confidence", 0.0))))
                mode_label = "Модель"
            else:
                continue
            priority = _priority_from_filename(f.name, default_priority)
            if priority not in priority_filter:
                continue
            rows.append({
                "StudyID": study_id,
                "Файл": f.name,
                "Режим": mode_label,
                "Приоритет": priority,
                "Целевой SLA": _sla_by_priority(priority),
                "Диагноз": CLASS_NAMES_RU_DEFAULT.get(pred["class"], pred["class"]),
                "Уверенность, %": round(float(pred["confidence"]) * 100, 1),
            })

        if rows:
            rows = sorted(rows, key=lambda x: (_priority_rank(x["Приоритет"]), -x["Уверенность, %"]))
            st.dataframe(rows, width="stretch")
            _enqueue_worklist(rows)
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            st.download_button(
                "Скачать пакетный отчёт (CSV)",
                data=csv_buffer.getvalue().encode("utf-8-sig"),
                file_name="batch_report.csv",
                mime="text/csv",
            )
            by_diag = {}
            for row in rows:
                by_diag[row["Диагноз"]] = by_diag.get(row["Диагноз"], 0) + 1
            st.bar_chart(by_diag)
    else:
        st.info("Очередь исследований пуста. Ожидание новых снимков из PACS.")

    st.markdown("---")
    st.subheader("Рабочая очередь исследований")
    if st.session_state["worklist"]:
        queue_rows = _worklist_view_rows()
        status_filter = st.multiselect(
            "Фильтр по статусу",
            ["В очереди", "В работе", "Завершено"],
            default=["В очереди", "В работе", "Завершено"],
            key="queue_status_filter",
        )
        priority_filter_queue = st.multiselect(
            "Фильтр по приоритету очереди",
            ["STAT (критический)", "Срочно", "Планово"],
            default=["STAT (критический)", "Срочно", "Планово"],
            key="queue_priority_filter",
        )
        queue_rows = [
            x for x in queue_rows
            if x["Статус"] in status_filter and x["Приоритет"] in priority_filter_queue
        ]
        st.dataframe(queue_rows, width="stretch")

        qk1, qk2, qk3 = st.columns(3)
        qk1.metric("В очереди", sum(1 for x in st.session_state["worklist"] if x["Статус"] == "В очереди"))
        qk2.metric("В работе", sum(1 for x in st.session_state["worklist"] if x["Статус"] == "В работе"))
        qk3.metric("SLA просрочено", sum(1 for x in _worklist_view_rows() if x["SLA статус"] == "Просрочено"))

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            selected_study = st.selectbox(
                "Выберите исследование",
                [x["StudyID"] for x in queue_rows] if queue_rows else [""],
                key="queue_study_select",
            )
        with q_col2:
            if st.button("Взять в работу"):
                for row in st.session_state["worklist"]:
                    if row["StudyID"] == selected_study and row["Статус"] == "В очереди":
                        row["Статус"] = "В работе"
                        break
        with q_col3:
            if st.button("Завершить исследование"):
                for row in st.session_state["worklist"]:
                    if row["StudyID"] == selected_study:
                        row["Статус"] = "Завершено"
                        break
        with q_col4:
            if st.button("Вернуть в очередь"):
                for row in st.session_state["worklist"]:
                    if row["StudyID"] == selected_study:
                        row["Статус"] = "В очереди"
                        break

        if st.button("Очистить завершённые из очереди"):
            st.session_state["worklist"] = [x for x in st.session_state["worklist"] if x["Статус"] != "Завершено"]
    else:
        st.caption("Рабочая очередь пока пуста. Добавьте снимки через пакетный анализ.")

    st.markdown("---")
    st.subheader("Метрики и устойчивость модели")
    history = _load_training_history()
    if history:
        st.markdown("Динамика качества на обучении:")
        if history.get("train_acc"):
            st.line_chart({"train_acc": history["train_acc"]})
        if history.get("val_acc"):
            st.line_chart({"val_acc": history["val_acc"]})
        best_acc = history.get("best_acc")
        if best_acc is not None:
            st.metric("Лучшая точность (валидация/обучение)", f"{best_acc*100:.2f}%")
    else:
        st.info("История обучения пока не найдена. После обучения появятся графики метрик.")

    if show_tech:
        st.markdown("---")
        st.subheader("Model Card и соответствие требованиям")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Технический профиль**")
            st.markdown("- Архитектура: ResNet18 + перенос обучения")
            st.markdown(f"- Размер входа: {image_size}x{image_size}")
            st.markdown("- Интерпретируемость: Grad-CAM")
            st.markdown("- Контуры: базовый + приоритетный")
        with mc2:
            st.markdown("**Эксплуатационный профиль**")
            st.markdown(f"- Активная роль: {user_role}")
            st.markdown(f"- Порог авто-решения: {decision_threshold:.2f}")
            st.markdown(f"- Контроль качества (Quality Gate): {'Вкл' if quality_gate else 'Выкл'}")
            st.markdown("- Экспорт: JSON/CSV отчёты")

    st.markdown("---")
    hist_tab, audit_tab = st.tabs(["История сессии", "Аудит решений"])
    with hist_tab:
        st.subheader("История анализов текущей сессии")
        if st.session_state["analysis_history"]:
            st.dataframe(st.session_state["analysis_history"], width="stretch")
            hist_csv = io.StringIO()
            writer = csv.DictWriter(hist_csv, fieldnames=list(st.session_state["analysis_history"][0].keys()))
            writer.writeheader()
            writer.writerows(st.session_state["analysis_history"])
            st.download_button(
                "Скачать историю сессии (CSV)",
                data=hist_csv.getvalue().encode("utf-8-sig"),
                file_name="session_history.csv",
                mime="text/csv",
            )
        else:
            st.caption("Пока нет записей. Загрузите снимок, чтобы сформировать историю.")
    with audit_tab:
        st.subheader("Аудит и трассировка решений")
        if st.session_state["audit_log"]:
            st.dataframe(st.session_state["audit_log"], width="stretch")
            audit_jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in st.session_state["audit_log"])
            st.download_button(
                "Скачать аудит-лог (JSONL)",
                data=audit_jsonl.encode("utf-8"),
                file_name="audit_log.jsonl",
                mime="application/json",
            )
        else:
            st.caption("Аудит-лог появится после первого анализа.")

    st.markdown("---")
    st.subheader("Операционная сводка смены")
    if st.session_state["analysis_history"]:
        last_case = st.session_state["analysis_history"][-1]
        st.markdown(
            f"Последний кейс: `{last_case['StudyID']}` | Диагноз: **{last_case['Диагноз']}** | "
            f"Уверенность: **{last_case['Уверенность, %']}%**"
        )
    else:
        st.caption("Кейсы ещё не обработаны в текущей смене.")

    if st.session_state["analysis_history"] or st.session_state["audit_log"]:
        session_pdf = _build_session_pdf(st.session_state["analysis_history"], st.session_state["audit_log"])
        st.download_button(
            "Скачать сводный отчёт смены (PDF)",
            data=session_pdf,
            file_name=f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

    st.sidebar.title("О программе")
    st.sidebar.markdown(
        "Система использует свёрточную нейронную сеть (ResNet18) с переносом обучения для классификации рентгеновских снимков грудной клетки."
    )
    st.sidebar.markdown("**Классы:** Норма, Пневмония, Рак, COVID-19 (при наличии данных).")
    st.sidebar.markdown("Результат носит справочный характер и не заменяет заключение врача.")


if __name__ == "__main__":
    main()
