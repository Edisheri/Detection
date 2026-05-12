"""Тесты is_likely_chest_xray — не требует весов модели."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.inference import is_likely_chest_xray


def _solid_gray(value: int = 128, size: int = 128) -> Image.Image:
    """Возвращает чисто-серое изображение (R=G=B), как рентген."""
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _noisy_gray(size: int = 128) -> Image.Image:
    """
    Симметричное серое изображение с тёмным фоном — имитация рентгена.
    Левая половина зеркалится в правую + минимальный шум,
    чтобы symmetry_diff оставался < 20 (порог — 50).
    """
    rng = np.random.default_rng(42)
    half = rng.integers(10, 160, (size, size // 2), dtype=np.uint8)
    noise = rng.integers(-4, 4, half.shape, dtype=np.int32)
    right = np.clip(np.fliplr(half).astype(np.int32) + noise, 10, 200).astype(np.uint8)
    full = np.concatenate([half, right], axis=1)
    arr = np.stack([full, full, full], axis=2)
    return Image.fromarray(arr)


def _colorful(size: int = 128) -> Image.Image:
    """Ярко-цветное изображение — должно быть отклонено фильтром."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 200   # R
    arr[:, :, 1] = 50    # G
    arr[:, :, 2] = 80    # B
    return Image.fromarray(arr)


def _tinted_gray(size: int = 128) -> Image.Image:
    """Тонированное ч/б (сепия) — не рентген."""
    rng = np.random.default_rng(7)
    base = rng.integers(30, 220, (size, size), dtype=np.uint8)
    arr = np.stack([
        np.clip(base.astype(int) + 20, 0, 255).astype(np.uint8),
        base,
        np.clip(base.astype(int) - 20, 0, 255).astype(np.uint8),
    ], axis=2)
    return Image.fromarray(arr)


class TestIsLikelyChestXray:
    def test_returns_expected_keys(self):
        result = is_likely_chest_xray(_solid_gray())
        assert set(result) == {"is_xray", "score", "details"}
        assert isinstance(result["is_xray"], bool)
        assert 0.0 <= result["score"] <= 1.0

    def test_noisy_gray_accepted_as_xray(self):
        result = is_likely_chest_xray(_noisy_gray())
        assert result["is_xray"] is True

    def test_colorful_image_rejected(self):
        result = is_likely_chest_xray(_colorful())
        assert result["is_xray"] is False

    def test_tinted_gray_rejected(self):
        result = is_likely_chest_xray(_tinted_gray())
        assert result["is_xray"] is False

    def test_score_range(self):
        for img in [_noisy_gray(), _colorful(), _tinted_gray()]:
            r = is_likely_chest_xray(img)
            assert 0.0 <= r["score"] <= 1.0

    def test_details_contains_all_fields(self):
        result = is_likely_chest_xray(_noisy_gray())
        d = result["details"]
        expected_fields = {
            "truly_grayscale", "low_saturation", "mostly_gray_pixels",
            "symmetric", "has_dark_bg", "contrast_ok", "aspect_ok",
            "per_pixel_diff", "mean_saturation", "close_px_fraction",
            "dark_fraction", "symmetry_diff", "contrast",
        }
        assert expected_fields.issubset(set(d.keys()))

    def test_per_pixel_diff_low_for_gray(self):
        result = is_likely_chest_xray(_noisy_gray())
        assert result["details"]["per_pixel_diff"] < 1.0

    def test_per_pixel_diff_high_for_color(self):
        result = is_likely_chest_xray(_colorful())
        assert result["details"]["per_pixel_diff"] > 5.0

    def test_accepts_pil_image_directly(self):
        img = _noisy_gray()
        r = is_likely_chest_xray(img)
        assert isinstance(r["is_xray"], bool)
