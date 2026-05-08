#!/usr/bin/env python3
"""
Самостоятельный скрипт оценки модели.
Сохраняет:
  reports/classification_report.txt
  reports/classification_report.json
  reports/confusion_matrix.png
  reports/metrics_summary.json
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import VAL_DIR, WEIGHTS_PATH, METRICS_DIR, IMAGE_SIZE
from src.dataset import ChestXRayDataset, get_transforms
from src.inference import load_model
from src.metrics import (
    evaluate_model, save_classification_report, save_confusion_matrix_png,
    save_metrics_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default=str(VAL_DIR))
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default=str(METRICS_DIR))
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        print(f"Validation directory not found: {val_dir}")
        sys.exit(1)

    if not Path(args.weights).exists():
        print(f"Weights not found: {args.weights}")
        sys.exit(1)

    model, class_names, image_size, device = load_model(args.weights)
    print(f"Устройство: {device}")
    print(f"Классы: {class_names}")

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    val_ds = ChestXRayDataset(
        str(val_dir),
        transform=get_transforms(image_size or IMAGE_SIZE, train=False),
        image_size=image_size or IMAGE_SIZE,
        class_to_idx=class_to_idx,
    )
    if len(val_ds) == 0:
        print("Нет данных для валидации.")
        sys.exit(1)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Validation samples: {len(val_ds)}")

    evaluation = evaluate_model(model, val_loader, device, class_names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_classification_report(evaluation, out_dir / "classification_report.txt")
    save_classification_report(evaluation, out_dir / "classification_report.json", as_json=True)
    save_confusion_matrix_png(evaluation, out_dir / "confusion_matrix.png")
    save_metrics_summary(evaluation, out_dir / "metrics_summary.json")

    print()
    print(f"Accuracy:        {evaluation['accuracy']:.4f}")
    print(f"Macro Precision: {evaluation['macro']['precision']:.4f}")
    print(f"Macro Recall:    {evaluation['macro']['recall']:.4f}")
    print(f"Macro F1:        {evaluation['macro']['f1']:.4f}")
    print()
    print(f"Метрики сохранены в {out_dir}")


if __name__ == "__main__":
    main()
