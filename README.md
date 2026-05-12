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

Актуальные значения после последнего `scripts/evaluate.py` — в **`reports/metrics_summary.json`** (папка создаётся при обучении/оценке).

Пример для сбалансированной валидации (по 500 снимков на класс): **Accuracy ≈ 79%**, **Macro F1 ≈ 78%** — см. JSON и графики в `reports/diagrams/` (скрипт `scripts/generate_metrics_visual.py`).

---

## Документация

- **[docs/](docs/)** — архитектура, сценарий видео для руководителя с **проходом по коду**, оценка структуры репозитория.

---

## Структура проекта

```
Detection/
├── app.py                 # Точка входа Streamlit (стили + lungdx.ui_main.run)
├── lungdx/                # UI-слой: клинические правила, PDF, сценарий run()
├── train.py               # Обучение ResNet18
├── config.py              # Пути, классы, гиперпараметры по умолчанию
├── run_app.py             # Запуск UI (CPU, безопасно для Streamlit на Windows)
├── pytest.ini             # Настройки pytest
├── tests/                 # Тесты (клиника, PDF)
├── requirements.txt
│
├── docs/                  # Документация (архитектура, сценарий видео)
├── src/                   # ML-ядро
│   ├── model.py           # ResNet18 + головка классификации
│   ├── dataset.py         # Dataset, transforms
│   ├── inference.py       # load_model, predict_image, Grad-CAM, X-ray фильтр
│   └── metrics.py         # Метрики, confusion matrix → reports/
│
├── scripts/               # CLI (не часть веб-рантайма)
│   ├── evaluate.py
│   ├── rebalance_dataset.py
│   ├── check_negatives.py
│   ├── generate_metrics_visual.py
│   ├── generate_diagrams.py
│   ├── generate_report.py
│   ├── generate_fixed_thesis_report.py
│   └── download_dataset.py
│
├── weights/               # best_model.pt, class_names.json, training_history.json (часть в .gitignore)
├── reports/               # метрики и PNG (генерируются, в .gitignore)
└── data/                  # датасет (в .gitignore)
    └── data/chest_xray/
        ├── train/
        └── val/
```

Отладочные черновики, не входящие в проект, могут храниться **вне репозитория**, например в `PycharmProjects/Detection_external_archive/`.

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python train.py --epochs 20 --batch-size 64
```

Пути к `train`/`val` берутся из `config.py` (или передайте `--train-dir` / `--val-dir`).

> Для GPU-ускорения установите CUDA-версию PyTorch:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

### 3. Запуск приложения

```bash
python run_app.py
# или напрямую:
streamlit run app.py --server.fileWatcherType none
```

Приложение откроется по адресу: **http://localhost:8501**

### 4. Тесты

```bash
pytest
```

Проверяют клинические правила (`lungdx/clinical.py`) и генерацию PDF без загрузки весов модели.

### 5. Оценка модели

```bash
python scripts/evaluate.py
```

Сохраняет в `reports/`: confusion_matrix.png, classification_report.txt, metrics_summary.json.

### 6. Проверка на негативных примерах

```bash
python scripts/check_negatives.py --dir /path/to/non-xray-images
```

### 7. Генерация отчёта (.docx)

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

Структура: подпапки классов в `train/` и `val/`. Конкретные объёмы зависят от вашей сборки; после **`scripts/rebalance_dataset.py`** типична **сбалансированная** валидация (например, по **500** снимков на класс). Точные числа — подсчёт файлов в каталогах или комментарий в скрипте балансировки.

---

## Лицензия

Для учебных и исследовательских целей.
