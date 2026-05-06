"""Inference for lung disease classification."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms
from src.model import build_model


def load_model(weights_path: str, device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    num_classes = ckpt.get("num_classes", 3)
    class_names = ckpt.get("class_names", ["Normal", "Pneumonia", "COVID-19"])
    image_size = ckpt.get("image_size", 224)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, image_size, device


def is_likely_chest_xray(image_or_path):
    """
    Простейшая эвристика, чтобы отличать рентген грудной клетки от произвольных фото.
    Используется только для того, чтобы не ставить диагноз на «левых» картинках.
    """
    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB")

    arr = np.array(img.resize((256, 256)))

    # 1) Картина должна быть практически чёрно‑белой (X‑ray обычно серый)
    flat = arr.reshape(-1, 3).astype("float32")
    ch_means = flat.mean(axis=0)
    color_spread = abs(ch_means[0] - ch_means[1]) + abs(ch_means[1] - ch_means[2])
    grayscale_like = color_spread < 15.0

    # 2) Центр должен быть светлее углов (лёгкие в середине, края темнее)
    g = arr.mean(axis=2)
    h, w = g.shape
    center = g[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    corners = np.concatenate(
        [
            g[: h // 4, : w // 4].ravel(),
            g[: h // 4, 3 * w // 4 :].ravel(),
            g[3 * h // 4 :, : w // 4].ravel(),
            g[3 * h // 4 :, 3 * w // 4 :].ravel(),
        ]
    ).mean()
    center_brighter = (center - corners) > 3.0

    return bool(grayscale_like and center_brighter)


def predict_image(model, image_path_or_pil, class_names, image_size, device):
    """
    Run prediction on a single image.
    image_path_or_pil: path (str/Path) or PIL.Image
    Returns: dict with class name, index, probabilities and flags.
    """
    transform = get_transforms(image_size, train=False)
    if isinstance(image_path_or_pil, (str, Path)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    looks_like_xray = is_likely_chest_xray(image)

    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())
    return {
        "class": class_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "confidence": float(probs[pred_idx]),
        "is_xray": looks_like_xray,
    }
