#!/usr/bin/env python3
"""
Проверка устойчивости системы на негативных примерах.

Прогоняет все изображения из заданной папки (по умолчанию data/negatives)
и фиксирует:
  1) сколько изображений было корректно отсеяно эвристикой is_likely_chest_xray
  2) какие классы и с какой уверенностью предсказывает модель,
     если эвристику отключить (для оценки переуверенности модели).

Результат: reports/negative_test_report.json
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import WEIGHTS_PATH, METRICS_DIR, BASE_DIR
from src.inference import load_model, predict_image, is_likely_chest_xray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(BASE_DIR / "data" / "negatives"))
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH))
    parser.add_argument("--out", type=str, default=str(METRICS_DIR / "negative_test_report.json"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Папка с негативными примерами не найдена: {input_dir}")
        print("Создайте её и положите туда фото не-рентген объектов (кот, пейзаж, машина и т.п.).")
        sys.exit(1)

    if not Path(args.weights).exists():
        print(f"Weights not found: {args.weights}")
        sys.exit(1)

    model, class_names, image_size, device = load_model(args.weights)

    files = [
        p for p in input_dir.rglob("*.*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ]

    if not files:
        print("В папке нет изображений.")
        sys.exit(1)

    rows = []
    rejected = 0
    for path in files:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            rows.append({"file": str(path), "error": str(e)})
            continue

        xray_check = is_likely_chest_xray(image)
        prediction = predict_image(model, image, class_names, image_size, device)
        if not xray_check["is_xray"]:
            rejected += 1

        rows.append({
            "file": str(path),
            "is_xray_heuristic": xray_check["is_xray"],
            "xray_score": xray_check["score"],
            "model_prediction": prediction["class"],
            "model_confidence": prediction["confidence"],
            "all_probabilities": prediction["probabilities"],
        })

    summary = {
        "total": len(files),
        "rejected_by_heuristic": rejected,
        "rejection_rate": rejected / len(files) if files else 0.0,
        "results": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Всего изображений: {len(files)}")
    print(f"Отсеяно эвристикой: {rejected} ({summary['rejection_rate']*100:.1f}%)")
    print(f"Отчёт: {out_path}")


if __name__ == "__main__":
    main()
