"""Конфигурация системы диагностики лёгочных заболеваний."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CHEST_XRAY_CANDIDATES = [
    DATA_DIR / "chest_xray",
    DATA_DIR / "data" / "chest_xray",
]


def _resolve_chest_xray_dir() -> Path:
    for candidate in CHEST_XRAY_CANDIDATES:
        if (candidate / "train").exists():
            return candidate
    return CHEST_XRAY_CANDIDATES[0]


CHEST_XRAY_DIR = _resolve_chest_xray_dir()
TRAIN_DIR = CHEST_XRAY_DIR / "train"
VAL_DIR = CHEST_XRAY_DIR / "val"

WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best_model.pt"
METRICS_DIR = BASE_DIR / "reports"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4

CLASS_NAMES = ["Cancer", "COVID-19", "Normal", "Pneumonia", "Tuberculosis"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_NAMES_RU = {
    "Normal": "Норма",
    "Pneumonia": "Пневмония",
    "COVID-19": "COVID-19",
    "Cancer": "Рак лёгкого",
    "Tuberculosis": "Туберкулёз",
}

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
