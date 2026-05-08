#!/usr/bin/env python3
"""Обучение классификатора лёгочных заболеваний на реальных данных."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAIN_DIR, VAL_DIR, WEIGHTS_DIR, WEIGHTS_PATH, METRICS_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
)
from src.dataset import ChestXRayDataset, get_transforms
from src.model import build_model
from src.metrics import (
    evaluate_model, save_classification_report, save_confusion_matrix_png,
    save_metrics_summary,
)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / max(len(loader), 1), correct / total


def compute_class_weights(samples, num_classes: int) -> torch.Tensor:
    """Сбалансированные веса классов для CrossEntropyLoss (несбалансированные данные)."""
    counts = Counter(label for _, label in samples)
    weights = []
    total = sum(counts.values())
    for i in range(num_classes):
        c = counts.get(i, 0)
        if c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default=str(TRAIN_DIR))
    parser.add_argument("--val-dir", type=str, default=str(VAL_DIR))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--output", type=str, default=str(WEIGHTS_PATH))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-per-class", type=int, default=0,
                        help="Если > 0 — ограничить количество изображений на класс в train.")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        sys.exit(1)

    train_ds = ChestXRayDataset(
        str(train_dir),
        transform=get_transforms(args.image_size, train=True),
        image_size=args.image_size,
        max_per_class=args.max_train_per_class or None,
    )
    if len(train_ds) == 0:
        print("No images found in train directory.")
        sys.exit(1)

    num_classes = len(train_ds.class_to_idx)
    class_names = list(train_ds.class_to_idx.keys())

    val_loader = None
    val_ds = None
    if val_dir.exists():
        val_ds = ChestXRayDataset(
            str(val_dir),
            transform=get_transforms(args.image_size, train=False),
            image_size=args.image_size,
            class_to_idx=train_ds.class_to_idx,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    batch_size = min(args.batch_size, len(train_ds)) if len(train_ds) > 0 else 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    print(f"Классы: {class_names}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}")

    class_weights = compute_class_weights(train_ds.samples, num_classes).to(device)
    print(f"Веса классов (для CrossEntropy): {class_weights.cpu().tolist()}")

    model = build_model(num_classes=num_classes, pretrained=True, freeze_backbone=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "image_size": args.image_size,
                }, output_path)
        else:
            if train_acc > best_acc:
                best_acc = train_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "image_size": args.image_size,
                }, output_path)
        scheduler.step()
        log_line = f"Epoch {epoch}: train loss={train_loss:.4f} acc={train_acc:.4f}"
        if val_loader:
            log_line += f" | val loss={val_loss:.4f} acc={val_acc:.4f}"
        print(log_line)

    if not output_path.exists():
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "image_size": args.image_size,
        }, output_path)

    meta_path = output_path.parent / "class_names.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    history_path = output_path.parent / "training_history.json"
    history["best_acc"] = best_acc
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if val_loader:
        print("Финальная оценка на валидации...")
        ckpt = torch.load(output_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        evaluation = evaluate_model(model, val_loader, device, class_names)
        save_classification_report(evaluation, METRICS_DIR / "classification_report.txt")
        save_classification_report(evaluation, METRICS_DIR / "classification_report.json", as_json=True)
        save_confusion_matrix_png(evaluation, METRICS_DIR / "confusion_matrix.png")
        save_metrics_summary(evaluation, METRICS_DIR / "metrics_summary.json", history=history)
        print(f"Метрики сохранены в {METRICS_DIR}")

    print(f"Модель сохранена: {output_path}")
    print(f"Классы: {class_names}")


if __name__ == "__main__":
    main()
