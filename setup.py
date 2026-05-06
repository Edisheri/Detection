#!/usr/bin/env python3
"""
Автоматическая подготовка данных и обучение при первом запуске.
Скачивает реальные рентген-снимки с открытого репозитория или создаёт демо-данные.
"""
import csv
import io
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

from PIL import Image


def _safe_filename(name: str, fallback: str = "img") -> str:
    """Оставить только безопасные символы в имени файла."""
    base = Path(name).name
    base = re.sub(r'[^\w\s\-\.]', '_', base)[:100]
    return base or fallback + ".jpg"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "chest_xray"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
WEIGHTS_PATH = ROOT / "weights" / "best_model.pt"

BASE_URL = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master"
METADATA_URL = f"{BASE_URL}/metadata.csv"
IMAGES_BASE = f"{BASE_URL}/images"

# Минимум изображений для обучения
MIN_IMAGES_PER_CLASS = 5
FIRST_RUN_EPOCHS = 5


def _download(url: str, timeout: int = 30):
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except (URLError, OSError) as e:
        print(f"  Ошибка загрузки {url}: {e}")
        return None


def _map_finding_to_class(finding: str, clinical_notes: str):
    """Маппинг поля finding в наш класс."""
    if not finding:
        return None
    f = finding.strip().lower()
    notes = (clinical_notes or "").lower()
    if "covid-19" in f or "covid" in f:
        return "COVID-19"
    if "pneumonia" in f or "sars" in f:
        return "Pneumonia"
    if "no finding" in f or "normal" in f or "no thoracic" in notes or "no abnormality" in notes:
        return "Normal"
    return None


def download_real_data() -> bool:
    """Скачать реальные снимки из открытого репозитория COVID-19."""
    print("Загрузка метаданных...")
    raw = _download(METADATA_URL)
    if not raw:
        return False
    try:
        text = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception as e:
        print(f"  Ошибка разбора CSV: {e}")
        return False

    # Собираем по классам (COVID-19, Pneumonia, Normal)
    by_class = {"COVID-19": [], "Pneumonia": [], "Normal": []}
    for r in rows:
        finding = r.get("finding", "")
        notes = r.get("clinical_notes", "")
        cls = _map_finding_to_class(finding, notes)
        if cls and cls in by_class:
            folder = (r.get("folder") or "images").strip()
            filename = (r.get("filename") or "").strip()
            if filename and folder == "images":
                by_class[cls].append(filename)

    # Собираем до 25 снимков на класс (без дубликатов по имени файла)
    max_per_class = 25
    to_download = []
    for cls in ["COVID-19", "Pneumonia", "Normal"]:
        seen = set()
        for fn in by_class[cls]:
            if fn not in seen and len([x for x in to_download if x[0] == cls]) < max_per_class:
                seen.add(fn)
                to_download.append((cls, fn))

    if len(to_download) < MIN_IMAGES_PER_CLASS * 2:
        print("  Недостаточно записей в метаданных.")
        return False

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    for cls in by_class:
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / cls).mkdir(parents=True, exist_ok=True)

    from urllib.parse import quote
    downloaded = 0
    for i, (cls, filename) in enumerate(to_download):
        url = f"{IMAGES_BASE}/{quote(filename)}"
        data = _download(url)
        if not data:
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        # 80% train, 20% val
        is_val = (i % 5) == 0
        folder = VAL_DIR if is_val else TRAIN_DIR
        safe_name = _safe_filename(filename, f"img_{downloaded}")
        if not safe_name.lower().endswith((".jpg", ".jpeg", ".png")):
            safe_name += ".jpg"
        path = folder / cls / safe_name
        try:
            img.save(path)
        except Exception:
            path = folder / cls / f"img_{downloaded}.jpg"
            img.save(path)
        downloaded += 1
        if downloaded % 10 == 0:
            print(f"  Загружено {downloaded} снимков...")

    total = sum(1 for _ in TRAIN_DIR.rglob("*.*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
    classes_with_data = len([d for d in TRAIN_DIR.iterdir() if d.is_dir() and list(d.glob("*.*"))])
    if total >= MIN_IMAGES_PER_CLASS * 2 and classes_with_data >= 2:
        print(f"Данные загружены: {total} снимков в train, классов: {classes_with_data}.")
        return True
    return False


def create_synthetic_data() -> None:
    """Создать минимальный демо-набор для работы приложения (только PIL, без numpy)."""
    print("Создание демо-набора изображений...")
    # Цвета по классам, чтобы модель могла различать
    colors = {"Normal": (200, 200, 200), "Pneumonia": (120, 120, 140), "COVID-19": (80, 100, 120)}
    for split, base in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        for cls in ["Normal", "Pneumonia", "COVID-19"]:
            d = base / cls
            d.mkdir(parents=True, exist_ok=True)
            rgb = colors[cls]
            for i in range(MIN_IMAGES_PER_CLASS):
                # Небольшие вариации яркости
                shift = (i * 17) % 40
                rgb_i = tuple(max(0, min(255, c - 20 + shift)) for c in rgb)
                img = Image.new("RGB", (224, 224), color=rgb_i)
                img.save(d / f"{cls}_{i}.png")
    print("Демо-набор создан (train + val).")


def ensure_data() -> None:
    """Гарантировать наличие данных: скачать или создать демо."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_count = sum(1 for _ in TRAIN_DIR.rglob("*.*") if _.suffix.lower() in (".jpg", ".jpeg", ".png")) if TRAIN_DIR.exists() else 0
    if train_count >= MIN_IMAGES_PER_CLASS * 2:
        print("Данные уже есть, пропуск загрузки.")
        return
    if not download_real_data():
        create_synthetic_data()


def run_training() -> None:
    """Запуск обучения (короткое при первом запуске)."""
    print("Обучение модели (это может занять несколько минут)...")
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--epochs", str(FIRST_RUN_EPOCHS),
        "--batch-size", "16",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError("Обучение завершилось с ошибкой.")


def ensure_model() -> None:
    """Подготовить данные и модель, если их ещё нет."""
    if WEIGHTS_PATH.exists():
        print("Модель уже обучена.")
        return
    ensure_data()
    run_training()
    if not WEIGHTS_PATH.exists():
        raise RuntimeError("После обучения файл модели не найден.")


if __name__ == "__main__":
    ensure_model()
    print("Готово. Запустите приложение: streamlit run app.py")
