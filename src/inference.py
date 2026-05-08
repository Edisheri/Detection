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
        # В Streamlit-процессе принудительно используем CPU чтобы
        # избежать конфликтов CUDA DLL в дочернем веб-процессе Windows.
        device = torch.device("cpu")
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


def is_likely_chest_xray(image_or_path) -> dict:
    """
    Эвристика для отсеивания "не-рентген" изображений (фото кота, пейзаж и т.п.).
    Возвращает dict со score и булевым решением, чтобы UI мог показывать
    степень доверия проверке, а не только бинарный флаг.
    """
    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB")

    arr = np.array(img.resize((256, 256))).astype("float32")

    flat = arr.reshape(-1, 3)
    ch_means = flat.mean(axis=0)
    color_spread = abs(ch_means[0] - ch_means[1]) + abs(ch_means[1] - ch_means[2])
    grayscale_like = color_spread < 18.0

    gray = arr.mean(axis=2)
    h, w = gray.shape
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
    corners = np.concatenate([
        gray[: h // 4, : w // 4].ravel(),
        gray[: h // 4, 3 * w // 4:].ravel(),
        gray[3 * h // 4:, : w // 4].ravel(),
        gray[3 * h // 4:, 3 * w // 4:].ravel(),
    ]).mean()
    center_brighter = (center - corners) > 3.0

    contrast = float(gray.std())
    contrast_ok = 25.0 < contrast < 90.0

    edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    edges_ok = 1.5 < edges < 25.0

    score_components = [grayscale_like, center_brighter, contrast_ok, edges_ok]
    score = sum(1.0 for x in score_components if x) / len(score_components)
    is_xray = score >= 0.75

    return {
        "is_xray": bool(is_xray),
        "score": float(score),
        "details": {
            "grayscale_like": bool(grayscale_like),
            "center_brighter": bool(center_brighter),
            "contrast_ok": bool(contrast_ok),
            "edges_ok": bool(edges_ok),
            "color_spread": float(color_spread),
            "contrast": float(contrast),
            "edges": float(edges),
        },
    }


def predict_image(model, image_path_or_pil, class_names, image_size, device) -> dict:
    """Один реальный прогон модели по изображению (без demo-логики)."""
    transform = get_transforms(image_size, train=False)
    if isinstance(image_path_or_pil, (str, Path)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    xray_check = is_likely_chest_xray(image)

    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[pred_idx])

    return {
        "class": class_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "confidence": confidence,
        "is_xray": xray_check["is_xray"],
        "xray_score": xray_check["score"],
        "xray_details": xray_check["details"],
    }


def generate_gradcam(model, image_path_or_pil, class_names, image_size, device, target_class_idx=None):
    """
    Generate Grad-CAM heatmap and overlay for a single image.
    Returns dict with overlay image and raw heatmap.
    """
    transform = get_transforms(image_size, train=False)
    if isinstance(image_path_or_pil, (str, Path)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    x = transform(image).unsqueeze(0).to(device)
    model.eval()

    features = []
    gradients = []

    if not hasattr(model, "layer4"):
        raise ValueError("Grad-CAM is supported for ResNet-like models with layer4.")

    target_layer = model.layer4[-1]

    def _forward_hook(_, __, output):
        features.append(output)

    def _backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(_forward_hook)
    backward_handle = target_layer.register_full_backward_hook(_backward_hook)

    try:
        logits = model(x)
        pred_idx = int(logits.argmax(dim=1).item())
        class_idx = pred_idx if target_class_idx is None else int(target_class_idx)
        class_idx = max(0, min(class_idx, logits.shape[1] - 1))

        score = logits[0, class_idx]
        model.zero_grad(set_to_none=True)
        score.backward()

        fmap = features[0][0]      # [C, H, W]
        grads = gradients[0][0]    # [C, H, W]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * fmap).sum(dim=0).detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        orig = np.array(image)
        h, w = orig.shape[:2]
        # Resize cam with PIL (no cv2 dependency)
        cam_pil = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
        cam_resized = np.array(cam_pil).astype(np.float32) / 255.0
        # Jet colormap via numpy
        t = cam_resized
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
        heatmap = np.stack([r, g, b], axis=-1)
        heatmap = np.uint8(heatmap * 255)
        overlay = np.clip(0.55 * orig + 0.45 * heatmap, 0, 255).astype(np.uint8)

        return {
            "predicted_class": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
            "target_class": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
            "overlay": overlay,
            "heatmap": np.uint8(cam_resized * 255),
        }
    finally:
        forward_handle.remove()
        backward_handle.remove()
