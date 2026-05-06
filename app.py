#!/usr/bin/env python3
"""Streamlit web app for lung disease diagnosis."""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WEIGHTS_PATH, CLASS_NAMES_RU
from src.inference import load_model, predict_image

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

CLASS_NAMES_RU_DEFAULT = {"Normal": "Норма", "Pneumonia": "Пневмония", "COVID-19": "COVID-19"}


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


def main():
    st.title("🫁 Диагностика лёгочных заболеваний")
    st.markdown("Загрузите рентгеновский снимок грудной клетки для анализа с помощью нейронной сети.")

    model, class_names, image_size, device = get_model()
    if model is None:
        st.warning("Модель не найдена. Запустите один раз: **python run.py** — данные подгрузятся и модель обучится автоматически.")
        st.info("Для теста можно загрузить любое изображение — будет показан пример интерфейса.")
        class_names = ["Normal", "Pneumonia", "COVID-19"]
        image_size = 224
        device = torch.device("cpu")

    uploaded = st.file_uploader("Выберите рентгеновский снимок (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Загруженный снимок", width="stretch")
        with col2:
            if model is not None:
                with st.spinner("Анализ..."):
                    result = predict_image(model, image, class_names, image_size, device)

                # Сначала проверяем, что это вообще похоже на рентген грудной клетки
                if not result.get("is_xray", True):
                    st.warning("Изображение не похоже на рентгеновский снимок грудной клетки. Модель не будет ставить диагноз.")
                else:
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
            else:
                st.info("Модель не загружена. Обучите модель и перезапустите приложение.")

    st.sidebar.title("О программе")
    st.sidebar.markdown(
        "Система использует свёрточную нейронную сеть (ResNet18) с переносом обучения для классификации рентгеновских снимков грудной клетки."
    )
    st.sidebar.markdown("**Классы:** Норма, Пневмония, COVID-19 (при наличии данных).")
    st.sidebar.markdown("Результат носит справочный характер и не заменяет заключение врача.")


if __name__ == "__main__":
    main()
