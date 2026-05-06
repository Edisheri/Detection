#!/usr/bin/env python3
"""Training script for lung disease classifier."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAIN_DIR, VAL_DIR, WEIGHTS_DIR, WEIGHTS_PATH,
    IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
)
from src.dataset import ChestXRayDataset, get_transforms
from src.model import build_model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default=str(TRAIN_DIR))
    parser.add_argument("--val-dir", type=str, default=str(VAL_DIR))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--output", type=str, default=str(WEIGHTS_PATH))
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        print("Create data/chest_xray/train/ with subfolders: Normal, Pneumonia (and optionally COVID-19)")
        print("Or run: python scripts/download_dataset.py")
        sys.exit(1)

    train_ds = ChestXRayDataset(
        str(train_dir),
        transform=get_transforms(args.image_size, train=True),
        image_size=args.image_size,
    )
    if len(train_ds) == 0:
        print("No images found in train directory.")
        sys.exit(1)

    num_classes = len(train_ds.class_to_idx)
    class_names = list(train_ds.class_to_idx.keys())

    val_loader = None
    if val_dir.exists():
        val_ds = ChestXRayDataset(
            str(val_dir),
            transform=get_transforms(args.image_size, train=False),
            image_size=args.image_size,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    batch_size = min(args.batch_size, len(train_ds)) if len(train_ds) > 0 else 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, pretrained=True, freeze_backbone=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
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
        print(f"Epoch {epoch}: train loss={train_loss:.4f} acc={train_acc:.4f}", end="")
        if val_loader:
            print(f" | val loss={val_loss:.4f} acc={val_acc:.4f}")
        else:
            print()

    # Сохранить модель в конце, если ещё не сохранена (например, только train без val)
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
    print(f"Model saved to {output_path}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
