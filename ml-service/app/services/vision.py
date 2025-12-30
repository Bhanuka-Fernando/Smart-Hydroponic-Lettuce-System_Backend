# ml-service/app/services/vision.py

from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import models

from app.core.paths import DEEPLAB_CKPT

# -----------------------------
# Calibration (YOU MUST SET)
# -----------------------------
# cm per pixel for your fixed camera setup.
# Example: if 1 pixel = 0.05 cm => CM_PER_PIXEL = 0.05
CM_PER_PIXEL = 0.05

# Optional: if you always used a fixed training input size, set it here.
# If you trained with original size, leave as None.
MODEL_INPUT_SIZE = None  # e.g., (512, 512) or None




@lru_cache(maxsize=1)
def get_seg_model():
    ckpt = torch.load(DEEPLAB_CKPT, map_location="cpu")

    state = (
        ckpt.get("state_dict")
        or ckpt.get("model_state_dict")
        or ckpt.get("model")
        or ckpt
    )

    # Strip "module." if trained with DataParallel
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    num_classes = ckpt.get("num_classes", 2) if isinstance(ckpt, dict) else 2

    model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Fail fast if checkpoint does not match
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing[:10]} ...")
    # unexpected keys are usually okay, but you can also enforce if needed

    model.eval()
    return model


def _decode_image(img_bytes: bytes) -> np.ndarray:
    """Decode bytes -> BGR uint8 image"""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes (invalid image).")
    return bgr


def _preprocess_for_deeplab(bgr: np.ndarray) -> torch.Tensor:
    """
    Converts BGR uint8 -> RGB float tensor normalized with ImageNet stats,
    shape: (1, 3, H, W)
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if MODEL_INPUT_SIZE is not None:
        rgb = cv2.resize(rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)

    rgb = rgb.astype(np.float32) / 255.0

    # ImageNet normalization (standard for ResNet backbones)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W)
    return x


def _mask_from_logits(logits: torch.Tensor) -> np.ndarray:
    """
    logits: (1, C, H, W)
    returns binary mask (H, W) uint8 {0,1}
    Assumes class 1 = plant, class 0 = background
    """
    pred = torch.argmax(logits, dim=1)  # (1,H,W)
    mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    # Convert any non-zero class to plant (safe even if you had >2 classes)
    mask = (mask > 0).astype(np.uint8)
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove noise; keep the largest connected component."""
    if mask.sum() == 0:
        return mask

    # Morphological open/close
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m

    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    cleaned = (labels == largest_idx).astype(np.uint8)
    return cleaned


def _area_and_diameter_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    """
    mask: (H,W) uint8 {0,1}
    returns: (A_proj_cm2, D_proj_cm)
    Diameter computed as max distance across the contour (approx via minEnclosingCircle).
    """
    if mask.sum() == 0:
        return 0.0, 0.0

    # Area
    area_px = float(mask.sum())
    A_proj_cm2 = area_px * (CM_PER_PIXEL ** 2)

    # Diameter (use contour -> minEnclosingCircle for robustness)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return A_proj_cm2, 0.0

    cnt = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    diameter_px = float(2.0 * radius)
    D_proj_cm = diameter_px * CM_PER_PIXEL

    return A_proj_cm2, D_proj_cm


@torch.no_grad()
def get_proj_area_and_diam(img_bytes: bytes) -> Tuple[float, float]:
    """
    Main function used by routes.py
    Input: raw image bytes (top view)
    Output: projected area (cm^2), projected diameter (cm)
    """
    bgr = _decode_image(img_bytes)

    x = _preprocess_for_deeplab(bgr)
    model = get_seg_model()

    out = model(x)["out"]  # (1,C,H,W)
    mask = _mask_from_logits(out)
    mask = _clean_mask(mask)

    A_cm2, D_cm = _area_and_diameter_from_mask(mask)
    return float(A_cm2), float(D_cm)
