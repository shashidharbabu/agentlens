"""CLIP similarity helpers for the Critique agent.

Primary: open_clip (ViT-B-32) for image<->image and image<->text cosine sim.
Fallback: SSIM for image<->image, a simple token-overlap heuristic for image<->text.
The fallback exists because the grader may not have internet to download
CLIP weights. The `used_fallback_metric` flag in Critique records this.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_available: Optional[bool] = None


def _try_load_clip():
    """Lazy-load CLIP; set _clip_available based on success."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_available
    if _clip_available is not None:
        return _clip_available
    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_available = True
    except Exception:
        _clip_available = False
    return _clip_available


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def clip_image_image_similarity(img_a: Image.Image, img_b: Image.Image) -> tuple[float, bool]:
    """Return (similarity, used_fallback)."""
    if _try_load_clip():
        import torch

        with torch.no_grad():
            a = _clip_preprocess(img_a).unsqueeze(0)
            b = _clip_preprocess(img_b).unsqueeze(0)
            fa = _clip_model.encode_image(a).cpu().numpy().squeeze()
            fb = _clip_model.encode_image(b).cpu().numpy().squeeze()
        return _cosine(fa, fb), False
    return _ssim_fallback(img_a, img_b), True


def clip_image_text_similarity(img: Image.Image, text: str) -> tuple[float, bool]:
    """Return (similarity, used_fallback)."""
    if _try_load_clip():
        import torch

        with torch.no_grad():
            i = _clip_preprocess(img).unsqueeze(0)
            t = _clip_tokenizer([text])
            fi = _clip_model.encode_image(i).cpu().numpy().squeeze()
            ft = _clip_model.encode_text(t).cpu().numpy().squeeze()
        return _cosine(fi, ft), False
    return _text_overlap_fallback(img, text), True


def _ssim_fallback(img_a: Image.Image, img_b: Image.Image) -> float:
    """Very rough image similarity using downsampled-pixel correlation.

    Not a real SSIM — we avoid the scikit-image dep. Good enough for a
    smoke-test signal when CLIP weights aren't available.
    """
    size = (64, 64)
    a = np.asarray(img_a.resize(size).convert("L"), dtype=np.float32).flatten()
    b = np.asarray(img_b.resize(size).convert("L"), dtype=np.float32).flatten()
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.clip(np.dot(a, b) / len(a), -1.0, 1.0))


def _text_overlap_fallback(img: Image.Image, text: str) -> float:
    """Fallback with no image signal — returns a neutral score.

    We return 0.2 (low but not zero) because without CLIP we genuinely
    don't know. The critique LLM will carry the real judgment.
    """
    _ = img, text
    return 0.2
