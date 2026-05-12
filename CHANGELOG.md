# Changelog

## [1.0.0] — 2026-05

### Added
- **`lungdx/`** — UI-пакет: клинические правила (`clinical.py`), PDF (`pdf_export.py`),
  стили, очередь, сессия, отрисовка результата, основной сценарий (`ui_main.run`).
- **`tests/`** — тесты pytest: клинические правила, PDF-генерация, X-ray фильтр.
- **`pyproject.toml`** — единый конфиг проекта (метаданные + pytest + ruff).
- **`.github/workflows/ci.yml`** — CI: установка зависимостей + pytest.
- **`CHANGELOG.md`** — журнал изменений.

### Changed
- **`app.py`** — тонкая точка входа (20 строк): `set_page_config` → `inject_styles()` → `run()`.
- **`docs/VIDEO_DEFENSE_SCRIPT.md`** — полный сценарий видеозащиты от А до Я.
- **`docs/REPO_QUALITY.md`** — обновлён под новую структуру.
- **Model Card** в `lungdx/ui_main.py` — исправлен: CosineAnnealingLR вместо StepLR,
  параметры берутся из `config.py`.

### Fixed
- Монолитный `app.py` (~860 строк) разбит на тестируемые модули.
- Устаревший StepLR в описании модели заменён на CosineAnnealingLR.

## [0.9.0] — 2026-04

### Added
- Пакетный анализ снимков + фильтрация по приоритету.
- Рабочая очередь STAT / Срочно / Планово с SLA и дедлайнами.
- Grad-CAM без OpenCV (pure PIL + NumPy).
- Quality Gate: блокировка авто-решения при плохом снимке.
- PDF-отчёт по снимку и сводный отчёт смены.
- Аудит-лог всех решений (JSONL).
- `is_likely_chest_xray` — эвристика: 7 признаков (насыщенность, ppd, симметрия, контраст…).

## [0.8.0] — 2026-03

### Added
- ResNet18 + Transfer Learning (ImageNet), 5 классов.
- Обучение: `train.py` с CosineAnnealingLR, class-weighted loss, сохранение `best_model.pt`.
- `scripts/evaluate.py`, `scripts/check_negatives.py`, `scripts/rebalance_dataset.py`.
- `src/metrics.py`: confusion matrix, classification report, `metrics_summary.json`.
