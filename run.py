#!/usr/bin/env python3
"""
Единая точка входа: при первом запуске автоматически подготавливаются
данные и обучается модель, затем запускается веб-приложение.
Запуск: python run.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "best_model.pt"


def main():
    if not WEIGHTS_PATH.exists():
        print("Первый запуск: подготовка данных и обучение модели...")
        try:
            from setup import ensure_model
            ensure_model()
        except Exception as e:
            print(f"Ошибка инициализации: {e}")
            sys.exit(1)
    sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    main()
