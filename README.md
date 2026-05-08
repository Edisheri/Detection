# LungDx Pro — Clinical Lung Disease Diagnosis Platform

Клиническая платформа автоматической диагностики лёгочных заболеваний на основе нейронной сети ResNet18.

> Магистерская дипломная работа  
> Тема: «Разработка программного обеспечения для диагностики лёгочных заболеваний с использованием нейронных сетей»

---

## Возможности

- **Классификация** рентгеновских снимков по 5 классам: Норма, Пневмония, COVID-19, Рак лёгкого, Туберкулёз
- **Grad-CAM** — тепловая карта внимания нейросети (интерпретируемость)
- **Quality Gate** — автоматический контроль качества снимка
- **Пакетный анализ** с сортировкой по приоритету и экспортом CSV
- **Рабочая очередь** исследований с приоритетами (STAT / Срочно / Планово) и SLA
- **PDF / JSON отчёты** по каждому снимку и сводный отчёт смены
- **Аудит-лог** всех решений системы (JSONL)
- **Фильтрация не-рентгеновских изображений** (эвристика + score)

## Метрики модели

| Метрика          | Значение |
|------------------|----------|
| Accuracy         | 98.87%   |
| Macro F1-score   | 78.67%   |
| Macro Precision  | 80.00%   |
| Macro Recall     | 80.00%   |

Обучение: ResNet18 + Transfer Learning, 10 эпох, GPU NVIDIA RTX 4060 (~22 мин), 13 224 снимков.

---

## Структура проекта

```
lungdx-pro/
├── app.py                   # Streamlit веб-приложение (клинический интерфейс)
├── train.py                 # Обучение модели
├── config.py                # Конфигурация путей и гиперпараметров
├── requirements.txt         # Зависимости Python
├── run_app.py               # Запуск приложения (CPU-режим)
│
├── src/                     # Основные модули
│   ├── model.py             # Архитектура ResNet18
│   ├── dataset.py           # DataLoader, аугментации
│   ├── inference.py         # Инференс, Grad-CAM, X-ray фильтр
│   └── metrics.py           # Accuracy, F1, confusion matrix
│
├── scripts/                 # Вспомогательные скрипты
│   ├── evaluate.py          # Оценка обученной модели
│   ├── check_negatives.py   # Тест на нерелевантных изображениях
│   ├── download_dataset.py  # Загрузка датасета
│   └── generate_report.py   # Генерация .docx отчёта
│
├── weights/                 # Артефакты модели
│   ├── best_model.pt        # Веса (не в git)
│   ├── class_names.json     # Имена классов
│   └── training_history.json# История обучения
│
├── reports/                 # Метрики (генерируются, не в git)
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── metrics_summary.json
│
└── data/                    # Датасет (не в git)
    └── data/chest_xray/
        ├── train/           # Cancer, COVID-19, Normal, Pneumonia, Tuberculosis
        └── val/
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python train.py \
  --train-dir data/data/chest_xray/train \
  --val-dir   data/data/chest_xray/val \
  --epochs 10 \
  --batch-size 64
```

> Для GPU-ускорения установите CUDA-версию PyTorch:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

### 3. Запуск приложения

```bash
python run_app.py
# или напрямую:
streamlit run app.py --server.fileWatcherType none
```

Приложение откроется по адресу: **http://localhost:8501**

### 4. Оценка модели

```bash
python scripts/evaluate.py
```

Сохраняет в `reports/`: confusion_matrix.png, classification_report.txt, metrics_summary.json.

### 5. Проверка на негативных примерах

```bash
python scripts/check_negatives.py --dir /path/to/non-xray-images
```

### 6. Генерация отчёта (.docx)

```bash
python scripts/generate_report.py
```

---

## Технологический стек

| Компонент    | Версия      |
|--------------|-------------|
| Python       | 3.14        |
| PyTorch      | 2.11.0      |
| Streamlit    | latest      |
| scikit-learn | ≥ 1.0       |
| Pillow       | ≥ 10.0      |
| NumPy        | ≥ 2.0       |
| matplotlib   | ≥ 3.5       |

---

## Датасет

Реальные клинические рентгеновские снимки грудной клетки (5 классов):

| Класс        | Train  | Val |
|--------------|--------|-----|
| COVID-19     | 3 358  | 26  |
| Рак лёгкого  | 3 875  | 9   |
| Норма        | 1 342  | 9   |
| Пневмония    | 3 875  | 9   |
| Туберкулёз   | 774    | 660 |
| **Итого**    | **13 224** | **713** |

---

## Лицензия

Для учебных и исследовательских целей.
