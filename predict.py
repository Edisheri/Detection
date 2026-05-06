#!/usr/bin/env python3
"""Command-line prediction for a single image."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WEIGHTS_PATH
from src.inference import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(description="Predict lung disease from chest X-ray image")
    parser.add_argument("image", type=str, help="Path to chest X-ray image")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH), help="Path to model weights")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Файл с весами модели не найден: {args.weights}")
        print("Сначала обучите модель: python train.py")
        sys.exit(1)

    model, class_names, image_size, device = load_model(args.weights)
    result = predict_image(model, args.image, class_names, image_size, device)

    if not result.get("is_xray", True):
        print("Изображение не похоже на рентгеновский снимок грудной клетки. Диагноз не ставится.")
        sys.exit(0)

    print("Класс:", result["class"])
    print("Уверенность:", f"{result['confidence']*100:.1f}%")
    print("Вероятности:", result["probabilities"])


if __name__ == "__main__":
    main()
