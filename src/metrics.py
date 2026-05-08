"""Расчёт и сохранение метрик качества классификации."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_model(model, loader, device, class_names) -> dict:
    """Прогон модели по DataLoader, расчёт метрик."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_targets.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    targets = np.array(all_targets)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    report_dict = classification_report(
        targets, preds, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    overall_acc = accuracy_score(targets, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0,
    )

    return {
        "class_names": class_names,
        "targets": targets.tolist(),
        "preds": preds.tolist(),
        "probabilities": probs.tolist(),
        "confusion_matrix": cm.tolist(),
        "report_dict": report_dict,
        "accuracy": float(overall_acc),
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
    }


def save_classification_report(evaluation: dict, output_path: Path, as_json: bool = False) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if as_json:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation["report_dict"], f, ensure_ascii=False, indent=2)
        return

    cls_names = evaluation["class_names"]
    targets = np.array(evaluation["targets"])
    preds = np.array(evaluation["preds"])
    text = classification_report(
        targets, preds, labels=list(range(len(cls_names))),
        target_names=cls_names, zero_division=0, digits=4,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_confusion_matrix_png(evaluation: dict, output_path: Path) -> None:
    cm = np.array(evaluation["confusion_matrix"], dtype=int)
    class_names = evaluation["class_names"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Матрица ошибок (validation)")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинно")

    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_summary(evaluation: dict, output_path: Path, history: Optional[dict] = None) -> None:
    summary = {
        "accuracy": evaluation["accuracy"],
        "macro": evaluation["macro"],
        "weighted": evaluation["weighted"],
        "per_class": {
            cls: evaluation["report_dict"].get(cls, {})
            for cls in evaluation["class_names"]
        },
        "confusion_matrix": evaluation["confusion_matrix"],
        "class_names": evaluation["class_names"],
    }
    if history:
        summary["training_history"] = history
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
