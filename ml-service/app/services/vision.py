# ml-service/app/services/vision.py
"""
Vision extractor:
- Input: RGB bytes + Depth bytes (typically 16-bit PNG depth)
- Output: projected area (cm^2), projected diameter (cm), median depth Z (m)

Also provides:
- get_plant_mask_u8(bgr) -> uint8 mask 0/255
- make_mask_applied_png(rgb_bytes) -> PNG bytes (background removed)
- make_mask_overlay_png(rgb_bytes) -> PNG bytes (green overlay)
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from app.core.config import settings
from app.core.paths import DEEPLAB_CKPT

SEG_INPUT_SIZE = 512


# -----------------------------
# Model loading + preprocessing
# -----------------------------


@lru_cache(maxsize=1)
def get_seg_model() -> torch.nn.Module:
    if not DEEPLAB_CKPT.exists():
        raise FileNotFoundError(f"Missing segmentation checkpoint: {DEEPLAB_CKPT}")

    # ✅ No aux head at inference (prevents aux_classifier size mismatch)
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=1,
        aux_loss=False,
    )

    state = torch.load(str(DEEPLAB_CKPT), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        # ✅ Drop aux head weights from checkpoint
        if nk.startswith("aux_classifier."):
            continue
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    # If you see missing keys here, your checkpoint is incomplete.
    if missing:
        raise RuntimeError(f"Seg checkpoint missing keys (first 15): {missing[:15]}")

    model.eval()
    return model




def _preprocess_for_deeplab(bgr_512: np.ndarray) -> torch.Tensor:
    """
    bgr_512: (512,512,3) uint8 BGR (OpenCV)
    returns: torch tensor (1,3,512,512) float32
    """
    rgb = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    if bool(getattr(settings, "USE_IMAGENET_NORM", False)):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std

    return x.float()


def _mask_from_logits(logits: torch.Tensor, thr: float) -> np.ndarray:
    """
    logits: torch tensor (1,C,H,W)
    returns mask (H,W) uint8 values 0/1
    Handles C=1 (sigmoid) or C=2 (softmax class1) robustly.
    """
    with torch.no_grad():
        if logits.ndim != 4:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        c = logits.shape[1]
        if c == 1:
            prob = torch.sigmoid(logits)[0, 0]
        elif c == 2:
            prob = torch.softmax(logits, dim=1)[0, 1]
        else:
            # fallback: treat argmax==1 as foreground
            pred = torch.argmax(logits, dim=1)[0]
            return (pred == 1).to(torch.uint8).cpu().numpy()

        mask = (prob > thr).to(torch.uint8).cpu().numpy()
        return mask  # 0/1


def _clean_mask(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: (H,W) 0/1
    Returns cleaned 0/1 mask with:
    - morphological open/close
    - largest connected component
    """
    if mask01 is None or mask01.size == 0:
        return mask01

    m = (mask01 * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    # Largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (m > 0).astype(np.uint8)

    # stats[0] is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    out = (labels == best_idx).astype(np.uint8)
    return out


def _decode_rgb(rgb_bytes: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode RGB image bytes.")
    return bgr


def _decode_depth(depth_bytes: bytes) -> np.ndarray:
    """
    Expects a depth image file that OpenCV can decode (commonly 16-bit PNG).
    Returns depth_raw as float32 (H,W) in meters.
    """
    d = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError("Could not decode Depth image bytes.")

    if d.ndim == 3:
        # if accidentally saved as 3-channel, take first channel
        d = d[:, :, 0]

    d = d.astype(np.float32)

    depth_scale = float(getattr(settings, "DEPTH_SCALE", 1.0))
    d_m = d * depth_scale
    return d_m


# -----------------------------
# Public helpers for mask images
# -----------------------------
def get_plant_mask_u8(bgr: np.ndarray, thr: Optional[float] = None) -> np.ndarray:
    """
    Returns uint8 mask in original image size with values 0/255.
    Uses the same segmentation pipeline as get_proj_area_and_diam().
    """
    H, W = bgr.shape[:2]

    bgr_512 = cv2.resize(bgr, (SEG_INPUT_SIZE, SEG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = _preprocess_for_deeplab(bgr_512)

    model = get_seg_model()
    with torch.no_grad():
        out = model(x)
        logits = out["out"] if isinstance(out, dict) and "out" in out else out

    if thr is None:
        thr = float(getattr(settings, "SEG_THRESHOLD", 0.5))

    mask_512 = _mask_from_logits(logits, thr)  # 0/1
    mask = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = _clean_mask(mask)  # 0/1
    return (mask * 255).astype(np.uint8)


def _encode_png(bgr_img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    return buf.tobytes()


def make_mask_applied_png(rgb_bytes: bytes) -> bytes:
    """
    Returns PNG bytes where background is black and only plant pixels remain.
    """
    bgr = _decode_rgb(rgb_bytes)
    mask_u8 = get_plant_mask_u8(bgr)  # 0/255
    masked = cv2.bitwise_and(bgr, bgr, mask=mask_u8)
    return _encode_png(masked)


def make_mask_overlay_png(rgb_bytes: bytes, alpha: float = 0.5) -> bytes:
    """
    Returns PNG bytes: original image with green overlay on plant mask.
    """
    bgr = _decode_rgb(rgb_bytes)
    mask_u8 = get_plant_mask_u8(bgr)  # 0/255

    overlay = bgr.copy()
    overlay[mask_u8 > 0] = (0, 255, 0)  # green in BGR
    out = cv2.addWeighted(overlay, float(alpha), bgr, 1.0 - float(alpha), 0.0)
    return _encode_png(out)


# -----------------------------
# Geometry: area + diameter
# -----------------------------
def _median_depth_in_mask(depth_m: np.ndarray, mask01: np.ndarray) -> float:
    """
    depth_m: (H,W) meters
    mask01: (H,W) 0/1
    Returns median depth (m) in plant region ignoring zeros.
    """
    vals = depth_m[(mask01 > 0) & (depth_m > 0)]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _projected_area_cm2(depth_m: np.ndarray, mask01: np.ndarray) -> float:
    """
    Approximates projected area in cm^2 using per-pixel metric area:
      pixel_area(m^2) = (Z/FX) * (Z/FY)
    Summed over mask pixels with valid depth.
    """
    FX = float(settings.FX)
    FY = float(settings.FY)

    z = depth_m
    ok = (mask01 > 0) & (z > 0)
    if not np.any(ok):
        return 0.0

    pixel_area_m2 = (z[ok] / FX) * (z[ok] / FY)
    area_m2 = float(np.sum(pixel_area_m2))
    return area_m2 * 1e4  # m^2 -> cm^2


def _feret_diameter_px(mask01: np.ndarray) -> float:
    """
    Returns an approximate max diameter in pixels using minAreaRect major axis.
    """
    m = (mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    return float(max(w, h))


def _diameter_cm(mask01: np.ndarray, Z_m: float) -> float:
    """
    Convert feret diameter pixels to cm using median depth.
    Uses average focal length scale.
    """
    if Z_m <= 0:
        return 0.0

    feret_px = _feret_diameter_px(mask01)
    if feret_px <= 0:
        return 0.0

    FX = float(settings.FX)
    FY = float(settings.FY)
    f = (FX + FY) / 2.0
    px_to_m = Z_m / f
    return float(feret_px * px_to_m * 100.0)  # m -> cm


def get_proj_area_and_diam(rgb_bytes: bytes, depth_bytes: bytes) -> Tuple[float, float, float]:
    """
    Main function used by your service.

    Returns:
      A_proj_cm2: float
      D_proj_cm:  float
      Z_m:        float (median depth in meters)
    """
    try:
        bgr = _decode_rgb(rgb_bytes)
        depth_m = _decode_depth(depth_bytes)

        H, W = bgr.shape[:2]
        if depth_m.shape[0] != H or depth_m.shape[1] != W:
            # align depth to RGB if needed
            depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

        # segmentation mask
        mask_u8 = get_plant_mask_u8(bgr)  # 0/255
        mask01 = (mask_u8 > 0).astype(np.uint8)

        if int(mask01.sum()) < 50:
            return 0.0, 0.0, 0.0

        Z_m = _median_depth_in_mask(depth_m, mask01)
        if Z_m <= 0:
            return 0.0, 0.0, 0.0

        A_proj_cm2 = _projected_area_cm2(depth_m, mask01)
        D_proj_cm = _diameter_cm(mask01, Z_m)

        if not np.isfinite(A_proj_cm2) or not np.isfinite(D_proj_cm):
            return 0.0, 0.0, 0.0

        return float(A_proj_cm2), float(D_proj_cm), float(Z_m)

    except Exception:
        # Safe fallback for API stability
        return 0.0, 0.0, 0.0
