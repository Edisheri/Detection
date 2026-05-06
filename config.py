"""Configuration for lung disease diagnosis."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHEST_XRAY_DIR = DATA_DIR / "chest_xray"
TRAIN_DIR = CHEST_XRAY_DIR / "train"
VAL_DIR = CHEST_XRAY_DIR / "val"
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best_model.pt"

# Model
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 3  # Normal, Pneumonia, COVID-19 (optional)

# Class names (order must match training)
CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]
CLASS_NAMES_RU = {"Normal": "Норма", "Pneumonia": "Пневмония", "COVID-19": "COVID-19"}

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
