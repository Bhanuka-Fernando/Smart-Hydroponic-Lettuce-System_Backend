# ml-service/app/services/vision.py

from functools import lru_cache
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision import models

from app.core.paths import DEEPLAB_CKPT
from app.core.config import settings  # FX, FY, DEPTH_SCALE, SEG_THRESHOLD, USE_IMAGENET_NORM


# Must match your Colab (you used 512x512 for segmentation)
SEG_INPUT_SIZE = 512


# -----------------------------
# Model
# -----------------------------
@lru_cache(maxsize=1)
def get_seg_model():
    ckpt = torch.load(DEEPLAB_CKPT, map_location="cpu")

    state = (
        ckpt.get("state_dict")
        or ckpt.get("model_state_dict")
        or ckpt.get("model")
        or ckpt
    )

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint format not supported (expected a state_dict-like dict).")

    # Strip "module." if trained with DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Infer num_classes from the classifier head weights if present
    # (works for 1-class logits too)
    head_key = "classifier.4.weight"
    if head_key in state:
        num_classes = int(state[head_key].shape[0])
    else:
        # fallback: if missing, assume 1 (binary logits)
        num_classes = int(getattr(settings, "SEG_NUM_CLASSES", 1))

    model = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing[:10]} ...")

    model.eval()
    return model


# -----------------------------
# IO helpers
# -----------------------------
def _decode_rgb(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode RGB image bytes.")
    return bgr


def _decode_depth(depth_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Could not decode depth image bytes.")
    if depth.dtype != np.uint16:
        # your dataset is uint16; if not, still handle safely
        depth = depth.astype(np.uint16)
    return depth


# -----------------------------
# Preprocess (match your Colab)
# -----------------------------
def _preprocess_for_deeplab(bgr_512: np.ndarray) -> torch.Tensor:
    """
    Colab behavior: RGB -> float32 /255, (optional) ImageNet norm OFF by default.
    Returns: (1,3,H,W) float tensor on CPU.
    """
    rgb = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if bool(settings.USE_IMAGENET_NORM):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W)
    return x


# -----------------------------
# Mask + geometry (match Colab)
# -----------------------------
def _mask_from_logits(logits: torch.Tensor, thr: float) -> np.ndarray:
    """
    - If logits has 1 channel: sigmoid + threshold (Colab)
    - Else: argmax (multi-class)
    Returns uint8 mask (H,W) with values {0,1}
    """
    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits)[0, 0]
        return (prob > float(thr)).cpu().numpy().astype(np.uint8)

    pred = torch.argmax(logits, dim=1)[0]
    return (pred > 0).cpu().numpy().astype(np.uint8)


def _clean_mask(mask01: np.ndarray) -> np.ndarray:
    """
    Match notebook style: open + close on a binary mask.
    """
    mask01 = np.ascontiguousarray(mask01.astype(np.uint8))
    if mask01.sum() == 0:
        return mask01

    m = (mask01 * 255).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)

    # notebook-like: small open then stronger close
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    return (m > 0).astype(np.uint8)


def _median_depth_m(depth_u16: np.ndarray, mask01: np.ndarray) -> float:
    """
    depth_u16: uint16 depth units
    Convert to meters using DEPTH_SCALE and take median over mask pixels (>0).
    """
    depth_m = depth_u16.astype(np.float32) * float(settings.DEPTH_SCALE)
    vals = depth_m[mask01 > 0]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _area_cm2_from_depth(depth_u16: np.ndarray, mask01: np.ndarray) -> float:
    """
    Colab method: per-pixel physical area from depth:
      pixel_w = Z / FX, pixel_h = Z / FY, area = sum(pixel_w * pixel_h)
    Then convert m^2 -> cm^2.
    """
    FX = float(settings.FX)
    FY = float(settings.FY)

    depth_m = depth_u16.astype(np.float32) * float(settings.DEPTH_SCALE)

    depth_plant = depth_m * mask01.astype(np.float32)
    # (ignore zeros)
    pixel_w_m = depth_plant / FX
    pixel_h_m = depth_plant / FY

    area_m2 = float((pixel_w_m * pixel_h_m).sum())
    return area_m2 * 1e4  # m^2 -> cm^2


def _feret_cm_from_mask(mask01: np.ndarray, Z_m: float) -> float:
    """
    Colab method: Feret diameter from minAreaRect on mask,
    then convert px -> cm using median depth:
      px_w_m = Z / FX  =>  cm = px * px_w_m * 100
    """
    if mask01.sum() == 0 or Z_m <= 0:
        return 0.0

    m = (mask01 * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0

    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    feret_px = float(max(w, h))

    px_w_m = Z_m / float(settings.FX)
    return feret_px * px_w_m * 100.0


# -----------------------------
# Public API (used by routes.py)
# -----------------------------
@torch.no_grad()
def get_proj_area_and_diam(
    rgb_bytes: bytes,
    depth_bytes: bytes,
    thr: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Input: RGB bytes + depth bytes (aligned)
    Output: A_proj_cm2, D_proj_cm, Z_m
    """
    bgr = _decode_rgb(rgb_bytes)
    depth = _decode_depth(depth_bytes)

    H, W = bgr.shape[:2]
    if depth.shape[:2] != (H, W):
        raise ValueError(f"RGB and Depth size mismatch: RGB={bgr.shape[:2]} Depth={depth.shape[:2]}")

    # 1) resize RGB to 512 for segmentation (exactly like Colab)
    bgr_512 = cv2.resize(bgr, (SEG_INPUT_SIZE, SEG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    x = _preprocess_for_deeplab(bgr_512)
    model = get_seg_model()
    logits = model(x)["out"]  # (1,C,512,512)

    if thr is None:
        thr = float(settings.SEG_THRESHOLD)

    # 2) mask in 512 space -> resize back to original
    mask_512 = _mask_from_logits(logits, thr)  # (512,512) 0/1
    mask = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = _clean_mask(mask)

    # 3) compute Z, area, feret
    Z_m = _median_depth_m(depth, mask)
    if Z_m <= 0 or mask.sum() == 0:
        return 0.0, 0.0, 0.0

    A_cm2 = _area_cm2_from_depth(depth, mask)
    D_cm = _feret_cm_from_mask(mask, Z_m)

    return float(A_cm2), float(D_cm), float(Z_m)
