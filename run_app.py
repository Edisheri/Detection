#!/usr/bin/env python3
"""Запуск Streamlit с принудительным CPU-режимом и без CUDA DLL.

Нужен потому что PyTorch 2.11+cu126 на Python 3.14 + Streamlit на Windows
вызывает access violation (0xC0000005) в дочернем процессе. Делаем CUDA
полностью невидимой ДО импорта torch.
"""
import os
import sys
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_DISABLE_CUDA"] = "1"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.fileWatcherType", "none",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    subprocess.run(cmd, env=os.environ.copy())
