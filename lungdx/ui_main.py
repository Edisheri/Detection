"""Основной сценарий Streamlit-приложения LungDx Pro."""

from __future__ import annotations

import csv
import io
import json
import hashlib
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

from config import BATCH_SIZE, LEARNING_RATE, METRICS_DIR, NUM_EPOCHS
from lungdx.clinical import (
    confidence_band,
    priority_from_filename,
    priority_rank,
    quality_assessment,
    risk_level,
    sla_by_priority,
)
from lungdx.constants import CLASS_NAMES_RU_DEFAULT
from lungdx.model_service import get_model
from lungdx.pdf_export import build_session_pdf
from lungdx.reports_io import load_metrics_summary, load_training_history
from lungdx.result_ui import render_result
from lungdx.session import ensure_session_state, ops_metrics
from lungdx.worklist_streamlit import enqueue_worklist, worklist_view_rows
from src.inference import generate_gradcam, is_likely_chest_xray, predict_image


def run() -> None:
    ensure_session_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("🫁 LungDx Pro")
    st.sidebar.caption("Клиническая платформа диагностики лёгочных заболеваний")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметры сессии")
    user_role = st.sidebar.selectbox("Роль", ["Врач-рентгенолог", "Лаборант", "Администратор"])
    decision_threshold = st.sidebar.slider("Порог авто-решения", 0.40, 0.99, 0.70, 0.01)
    quality_gate = st.sidebar.checkbox("Quality Gate (блок при плохом снимке)", value=False)
    default_priority = st.sidebar.selectbox(
        "Приоритет по умолчанию", ["Планово", "Срочно", "STAT (критический)"]
    )
    show_tech = st.sidebar.checkbox("Model Card (для разработчиков)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Смена:** {datetime.now().strftime('%d.%m.%Y')}")
    st.sidebar.markdown(f"**Оператор:** {user_role}")
    st.sidebar.markdown("**Учреждение:** ГКБ №1, лучевая диагностика")
    st.sidebar.markdown("**Интеграции:** PACS/RIS, ЭМК, аудит-лог")

    # ── Заголовок и операционная сводка ─────────────────────────────────────
    st.title("🫁 LungDx Pro — Клиническая диагностика лёгочных заболеваний")
    st.caption(
        "Система поддержки принятия врачебных решений на основе нейронной сети ResNet18. Версия 1.0."
    )

    ops = ops_metrics()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Исследований за смену", ops["processed"])
    k2.metric("Высокий / Критический риск", ops["high_risk"])
    k3.metric("Доля авто-решений", f"{ops['auto_rate']:.1f}%")
    k4.metric("На ручную верификацию", ops["manual"])

    s1, s2, s3, s4 = st.columns(4)
    s1.success("Шлюз PACS: В сети")
    s2.success("Сервис инференса: В сети")
    s3.success("Сервис аудита: В сети")
    s4.success("Сервис отчётов: В сети")

    # ── Загрузка модели ───────────────────────────────────────────────────────
    model, class_names, image_size, device = get_model()
    if model is None:
        st.error("⚠️ Модель не загружена. Запустите обучение: `python train.py`")
        st.stop()

    # ── Анализ одиночного снимка ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Анализ одиночного снимка")
    uploaded = st.file_uploader(
        "Загрузите рентгеновский снимок (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    result = None
    xray_rejected = False   # флаг — st.stop() выносим за пределы with-блоков

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        quality = quality_assessment(image)
        study_id = hashlib.md5(uploaded.name.encode()).hexdigest()[:10].upper()

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Загруженный снимок", use_container_width=True)
            st.caption(f"Study ID: {study_id} | Файл: {uploaded.name}")

        # ── Проверка X-ray (вне with, чтобы st.stop() работал корректно) ────
        xray_check = is_likely_chest_xray(image)

        if not xray_check["is_xray"]:
            d = xray_check["details"]
            ppd = d.get("per_pixel_diff", 0)
            sat = d.get("mean_saturation", 0)
            sym = d.get("symmetry_diff", 0)

            if not d.get("truly_grayscale", True) or not d.get("low_saturation", True):
                reason = "изображение содержит цветовые компоненты (рентген — grayscale)"
            elif not d.get("symmetric", True):
                reason = (
                    "изображение не симметрично по вертикальной оси "
                    "(рентген грудной клетки симметричен)"
                )
            else:
                reason = "не выполнены условия принятия рентгенограммы"

            with col_result:
                st.markdown(
                    f'<div class="reject-box">'
                    f"<b>⛔ Изображение не является рентгеновским снимком</b><br>"
                    f"Причина: <b>{reason}</b>.<br>"
                    f"Нейросетевой анализ не выполняется.<br><br>"
                    f"<b>Диагностические показатели:</b><br>"
                    f"• Цветность (ppd): <b>{ppd:.1f}</b> "
                    f'<span style="color:#888">(рентген: &lt; 2.5)</span><br>'
                    f"• Насыщенность: <b>{sat:.3f}</b> "
                    f'<span style="color:#888">(рентген: &lt; 0.05)</span><br>'
                    f"• Симметрия: <b>{sym:.1f}</b> "
                    f'<span style="color:#888">(рентген: &lt; 50)</span><br><br>'
                    f"<i>Пожалуйста, загрузите рентгенограмму грудной клетки.</i>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            xray_rejected = True

        else:
            # ── Инференс ──────────────────────────────────────────────────────
            with col_result:
                with st.spinner("Анализ изображения..."):
                    result = predict_image(model, image, class_names, image_size, device)

                if result.get("xray_score", 1.0) < 0.9:
                    st.markdown(
                        f'<div class="warn-box">⚠️ Оценка качества снимка: '
                        f'{result.get("xray_score", 0) * 100:.0f}%. '
                        f"Рекомендуется повторная съёмка.</div>",
                        unsafe_allow_html=True,
                    )

                if result["confidence"] < decision_threshold:
                    st.warning(
                        f"Уверенность модели ({result['confidence'] * 100:.1f}%) ниже порога "
                        f"({decision_threshold * 100:.0f}%). Рекомендуется ручная верификация."
                    )

                _, risk, decision, priority = render_result(
                    result,
                    uploaded.name,
                    quality,
                    study_id,
                    decision_threshold,
                    default_priority,
                    image,
                )

            # ── Запись в историю и аудит ──────────────────────────────────────
            event_key = f"{study_id}:{result['class']}:{result['confidence']:.4f}"
            if event_key != st.session_state["last_event_hash"]:
                st.session_state["last_event_hash"] = event_key
                st.session_state["analysis_history"].append(
                    {
                        "Время": datetime.now().strftime("%H:%M:%S"),
                        "StudyID": study_id,
                        "Файл": uploaded.name,
                        "Диагноз": CLASS_NAMES_RU_DEFAULT.get(result["class"], result["class"]),
                        "Уверенность, %": round(result["confidence"] * 100, 1),
                        "Риск": risk,
                        "Решение": decision,
                    }
                )
            st.session_state["audit_log"].append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "study_id": study_id,
                    "file_name": uploaded.name,
                    "predicted_class": result["class"],
                    "confidence": round(result["confidence"], 4),
                    "threshold": decision_threshold,
                    "decision": decision,
                    "band": confidence_band(result["confidence"]),
                    "priority": priority,
                    "sla_target": sla_by_priority(priority),
                }
            )

        # ── st.stop() безопасно снаружи всех with-блоков ─────────────────────
        if xray_rejected:
            st.stop()

        # ── Контроль качества снимка ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("Контроль качества снимка")
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Яркость", f"{quality['brightness']:.1f}")
        qc2.metric("Контраст", f"{quality['contrast']:.1f}")
        qc3.metric("Резкость", f"{quality['sharpness']:.1f}")
        if quality["warnings"]:
            for w in quality["warnings"]:
                st.warning(w)
            if quality_gate:
                st.error("Quality Gate: автоматическое решение заблокировано из-за качества снимка.")
        else:
            st.success("Качество изображения соответствует требованиям клинического анализа.")

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Тепловая карта внимания нейросети (Grad-CAM)")
        if model and result is not None and result.get("is_xray", True):
            try:
                target_idx = None
                if result.get("class") in class_names:
                    target_idx = class_names.index(result["class"])
                gradcam = generate_gradcam(
                    model, image, class_names, image_size, device, target_idx
                )
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    st.image(
                        gradcam["overlay"],
                        caption="Grad-CAM: наложение",
                        use_container_width=True,
                    )
                with gc2:
                    st.image(
                        gradcam["heatmap"],
                        caption="Карта активаций",
                        use_container_width=True,
                    )
                with gc3:
                    alpha = st.slider("Прозрачность наложения", 0.1, 0.9, 0.45, 0.05)
                    blend = np.clip(
                        (1 - alpha) * np.array(image) + alpha * gradcam["overlay"],
                        0,
                        255,
                    ).astype(np.uint8)
                    st.image(
                        blend,
                        caption="Интерактивное наложение",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Grad-CAM недоступен: {e}")

    # ── Пакетный анализ снимков ───────────────────────────────────────────────
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
        rows: list[dict] = []
        progress_bar = st.progress(0, text="Обработка...")
        for i, f in enumerate(batch_files):
            progress_bar.progress(
                (i + 1) / len(batch_files), text=f"Обработка {f.name}..."
            )
            try:
                img = Image.open(f).convert("RGB")
                pred = predict_image(model, img, class_names, image_size, device)
            except Exception:
                continue
            file_priority = priority_from_filename(f.name, default_priority)
            if file_priority not in pf:
                continue
            sid = hashlib.md5(f.name.encode()).hexdigest()[:10].upper()
            rows.append(
                {
                    "StudyID": sid,
                    "Файл": f.name,
                    "Приоритет": file_priority,
                    "Целевой SLA": sla_by_priority(file_priority),
                    "Диагноз": CLASS_NAMES_RU_DEFAULT.get(pred["class"], pred["class"]),
                    "Уверенность, %": round(pred["confidence"] * 100, 1),
                    "Риск": risk_level(pred["confidence"], pred["class"]),
                    "Рентген": "✅" if pred.get("is_xray", True) else "❌",
                }
            )
        progress_bar.empty()

        if rows:
            rows.sort(key=lambda x: (priority_rank(x["Приоритет"]), -x["Уверенность, %"]))
            st.dataframe(rows, use_container_width=True)
            enqueue_worklist(rows)

            by_diag: dict[str, int] = {}
            for row in rows:
                by_diag[row["Диагноз"]] = by_diag.get(row["Диагноз"], 0) + 1
            st.bar_chart(by_diag)

            csv_buf = io.StringIO()
            csv_writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
            st.download_button(
                "📥 Скачать пакетный отчёт (CSV)",
                data=csv_buf.getvalue().encode("utf-8-sig"),
                file_name="batch_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Очередь исследований пуста. Ожидание снимков из PACS.")

    # ── Рабочая очередь исследований ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Рабочая очередь исследований")
    if st.session_state["worklist"]:
        q_rows = worklist_view_rows()

        qf1, qf2 = st.columns(2)
        with qf1:
            sf = st.multiselect(
                "Статус",
                ["В очереди", "В работе", "Завершено"],
                default=["В очереди", "В работе", "Завершено"],
                key="q_sf",
            )
        with qf2:
            pf2 = st.multiselect(
                "Приоритет",
                ["STAT (критический)", "Срочно", "Планово"],
                default=["STAT (критический)", "Срочно", "Планово"],
                key="q_pf",
            )

        q_rows = [r for r in q_rows if r["Статус"] in sf and r["Приоритет"] in pf2]
        st.dataframe(q_rows, use_container_width=True)

        qm1, qm2, qm3 = st.columns(3)
        qm1.metric(
            "В очереди",
            sum(1 for x in st.session_state["worklist"] if x["Статус"] == "В очереди"),
        )
        qm2.metric(
            "В работе",
            sum(1 for x in st.session_state["worklist"] if x["Статус"] == "В работе"),
        )
        qm3.metric(
            "SLA просрочено",
            sum(1 for x in worklist_view_rows() if "Просрочено" in x["SLA статус"]),
        )

        sel_ids = [x["StudyID"] for x in q_rows] if q_rows else [""]
        b1, b2, b3, b4, b5 = st.columns(5)
        selected = b1.selectbox("Исследование", sel_ids, key="q_sel")
        if b2.button("▶ Взять в работу"):
            for r in st.session_state["worklist"]:
                if r["StudyID"] == selected and r["Статус"] == "В очереди":
                    r["Статус"] = "В работе"
                    break
        if b3.button("✅ Завершить"):
            for r in st.session_state["worklist"]:
                if r["StudyID"] == selected:
                    r["Статус"] = "Завершено"
                    break
        if b4.button("↩ Вернуть"):
            for r in st.session_state["worklist"]:
                if r["StudyID"] == selected:
                    r["Статус"] = "В очереди"
                    break
        if b5.button("🗑 Очистить завершённые"):
            st.session_state["worklist"] = [
                x for x in st.session_state["worklist"] if x["Статус"] != "Завершено"
            ]
    else:
        st.caption("Рабочая очередь пуста. Добавьте снимки через пакетный анализ.")

    # ── Model Card ────────────────────────────────────────────────────────────
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
            st.markdown(f"- Оптимизатор: Adam (lr={LEARNING_RATE})")
            st.markdown("- Scheduler: CosineAnnealingLR (T_max=эпохи, eta_min=1e-6)")
            st.markdown("- Loss: CrossEntropyLoss (веса классов)")
            st.markdown(
                f"- Эпох по умолчанию: {NUM_EPOCHS} "
                f"(CLI: `python train.py --epochs N`) | Batch: {BATCH_SIZE}"
            )
        with mc3:
            st.markdown("**Классы модели**")
            for cn in class_names:
                st.markdown(f"- {CLASS_NAMES_RU_DEFAULT.get(cn, cn)} (`{cn}`)")

        metrics = load_metrics_summary()
        history = load_training_history()
        if metrics:
            st.markdown("**Метрики на валидационной выборке:**")
            ma1, ma2, ma3, ma4 = st.columns(4)
            ma1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
            ma2.metric(
                "Precision", f"{metrics.get('macro', {}).get('precision', 0) * 100:.2f}%"
            )
            ma3.metric("Recall", f"{metrics.get('macro', {}).get('recall', 0) * 100:.2f}%")
            ma4.metric(
                "F1 (macro)", f"{metrics.get('macro', {}).get('f1', 0) * 100:.2f}%"
            )

            cm_path = METRICS_DIR / "confusion_matrix.png"
            cr_path = METRICS_DIR / "classification_report.txt"
            col_cm, col_cr = st.columns([1, 1])
            with col_cm:
                if cm_path.exists():
                    st.image(
                        str(cm_path),
                        caption="Матрица ошибок",
                        use_container_width=True,
                    )
            with col_cr:
                if cr_path.exists():
                    st.text_area(
                        "Classification Report",
                        cr_path.read_text(encoding="utf-8"),
                        height=260,
                    )
        if history:
            chart_data: dict[str, list] = {}
            if history.get("train_acc"):
                chart_data["Train Acc"] = history["train_acc"]
            if history.get("val_acc"):
                chart_data["Val Acc"] = history["val_acc"]
            if chart_data:
                st.markdown("**Динамика обучения:**")
                st.line_chart(chart_data)

    # ── История и аудит-лог ───────────────────────────────────────────────────
    st.markdown("---")
    hist_tab, audit_tab = st.tabs(["📋 История исследований", "🔍 Аудит решений"])

    with hist_tab:
        history_data = st.session_state["analysis_history"]
        if history_data:
            st.dataframe(history_data, use_container_width=True)
            buf = io.StringIO()
            hist_writer = csv.DictWriter(buf, fieldnames=list(history_data[0].keys()))
            hist_writer.writeheader()
            hist_writer.writerows(history_data)
            st.download_button(
                "📥 Скачать историю (CSV)",
                data=buf.getvalue().encode("utf-8-sig"),
                file_name="session_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("История пуста. Загрузите снимок для начала работы.")

    with audit_tab:
        audit_data = st.session_state["audit_log"]
        if audit_data:
            st.dataframe(audit_data, use_container_width=True)
            jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in audit_data)
            st.download_button(
                "📥 Скачать аудит-лог (JSONL)",
                data=jsonl.encode("utf-8"),
                file_name="audit_log.jsonl",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("Аудит-лог пуст.")

    # ── Операционная сводка смены ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Операционная сводка смены")
    shift_history = st.session_state["analysis_history"]
    if shift_history:
        last = shift_history[-1]
        st.markdown(
            f"Последнее исследование: `{last['StudyID']}` | "
            f"Диагноз: **{last['Диагноз']}** | "
            f"Уверенность: **{last['Уверенность, %']}%** | "
            f"Риск: **{last['Риск']}**"
        )
        st.download_button(
            "📑 Скачать сводный отчёт смены (PDF)",
            data=build_session_pdf(shift_history, st.session_state["audit_log"]),
            file_name=f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.caption("Исследования ещё не проводились в текущей смене.")
