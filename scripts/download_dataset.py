#!/usr/bin/env python3
"""
Download a sample chest X-ray dataset for training.
Creates folder structure; uses Kaggle API if available.
"""
import shutil
import zipfile
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "chest_xray"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# Sample dataset: COVID-19 Radiography Database (small subset) or similar
# Using a commonly used small dataset URL (example - may need update)
# Alternative: Kaggle "Chest X-Ray Images (Pneumonia)" - user must download manually
SAMPLE_ZIP_URL = (
    "https://github.com/ieee8023/covid-chestxray-dataset/archive/refs/heads/master.zip"
)
# Simpler: we create minimal structure and suggest user to add images
# Or use gdown for a Google Drive dataset


def create_demo_structure():
    """Create folder structure and a README for manual dataset placement."""
    for split in ["train", "val"]:
        base = DATA_DIR / split
        for cls in ["Normal", "Pneumonia"]:
            (base / cls).mkdir(parents=True, exist_ok=True)
    readme = DATA_DIR / "README.txt"
    readme.write_text(
        "Разместите снимки по папкам:\n"
        "  train/Normal/   - снимки без патологии\n"
        "  train/Pneumonia/ - снимки с пневмонией\n"
        "  val/Normal/, val/Pneumonia/ - для проверки\n\n"
        "Рекомендуемый датасет: Kaggle 'Chest X-Ray Images (Pneumonia)'\n"
        "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
        "После загрузки распакуйте и скопируйте train/val в data/chest_xray/\n",
        encoding="utf-8",
    )
    print(f"Создана структура папок в {DATA_DIR}")
    print("Добавьте изображения в train/Normal и train/Pneumonia (и val при необходимости).")
    print("Либо скачайте датасет с Kaggle: Chest X-Ray Images (Pneumonia).")


def try_download_kaggle_style():
    """Try to download using kaggle CLI if available."""
    try:
        import subprocess
        # Kaggle dataset: paultimothymooney/chest-xray-pneumonia
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia"],
            cwd=str(DATA_DIR),
            check=True,
        )
        zip_path = DATA_DIR / "chest-xray-pneumonia.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            # Dataset has chest_xray/train, chest_xray/val, chest_xray/test
            extracted = DATA_DIR / "chest_xray"
            if extracted.exists():
                for name in ["train", "val"]:
                    src = extracted / name
                    if src.exists():
                        dest = DATA_DIR / name
                        if dest.exists():
                            for cls_dir in src.iterdir():
                                if cls_dir.is_dir():
                                    dest_cls = dest / cls_dir.name
                                    dest_cls.mkdir(parents=True, exist_ok=True)
                                    for f in cls_dir.iterdir():
                                        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                                            shutil.copy2(f, dest_cls / f.name)
                        else:
                            shutil.copytree(src, dest)
                shutil.rmtree(extracted, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
            print("Датасет успешно загружен через Kaggle API.")
            return True
    except Exception as e:
        print(f"Kaggle загрузка недоступна: {e}")
    return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if try_download_kaggle_style():
        return
    create_demo_structure()


if __name__ == "__main__":
    main()
