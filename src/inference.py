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
    Эвристика для отсеивания "не-рентген" изображений.

    Надёжные признаки рентгена грудной клетки:
      1. Низкая насыщенность цвета (рентген — чёрно-белый по природе)
      2. Большая доля тёмных пикселей (чёрный фон)
      3. Умеренный контраст (не плоский, не слишком резкий)
      4. Допустимое соотношение сторон (рентген ≈ квадратный)
      5. Характерный диапазон яркости (тёмный, но с яркими областями)
    """
    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB")

    arr = np.array(img.resize((256, 256))).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = arr.mean(axis=2)

    # 1. Насыщенность цвета (главный критерий)
    #    Рентген почти чисто серый → насыщенность ≈ 0.
    #    Фото людей/животных/природы имеют высокую насыщенность.
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = np.where(max_ch > 1e-3, (max_ch - min_ch) / np.where(max_ch > 1e-3, max_ch, 1.0), 0.0)
    mean_sat = float(saturation.mean())
    low_saturation = mean_sat < 0.10          # цветные фото обычно > 0.20

    # 2. Доля тёмных пикселей — рентген имеет много чёрного фона
    dark_fraction = float((gray < 40).mean())
    has_dark_bg = dark_fraction > 0.10        # > 10% пикселей почти чёрные

    # 3. Контраст — рентген: std серого в диапазоне 30-90
    contrast = float(gray.std())
    contrast_ok = 28.0 < contrast < 95.0

    # 4. Соотношение сторон — рентген грудной клетки ~0.7–1.4
    w_px, h_px = img.size
    aspect = w_px / max(h_px, 1)
    aspect_ok = 0.60 < aspect < 1.55

    # 5. Средняя яркость — рентген в диапазоне 30-160 (темнее обычного фото)
    mean_brightness = float(gray.mean())
    brightness_ok = 20.0 < mean_brightness < 175.0

    checks = [low_saturation, has_dark_bg, contrast_ok, aspect_ok, brightness_ok]
    score = sum(1.0 for c in checks if c) / len(checks)

    # Обязательное условие: насыщенность ДОЛЖНА быть низкой.
    # Без него цветные фото могут набрать 4/5 по остальным критериям.
    is_xray = bool(score >= 0.60 and low_saturation)

    return {
        "is_xray": is_xray,
        "score": float(score),
        "details": {
            "low_saturation": bool(low_saturation),
            "has_dark_bg": bool(has_dark_bg),
            "contrast_ok": bool(contrast_ok),
            "aspect_ok": bool(aspect_ok),
            "brightness_ok": bool(brightness_ok),
            "mean_saturation": round(mean_sat, 3),
            "dark_fraction": round(dark_fraction, 3),
            "contrast": round(contrast, 1),
            "mean_brightness": round(mean_brightness, 1),
            "aspect": round(aspect, 2),
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
