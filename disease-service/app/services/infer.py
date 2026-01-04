import os
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv

import torch
import timm
from timm.data import resolve_model_data_config, create_transform
from app.services.annotate import draw_tipburn_boxes, pil_to_png_bytes
from ultralytics import YOLO

from app.services.health_score import compute_health_score

load_dotenv()

CLASSES = ["Bacterial", "Fungal", "Healthy", "K_Def", "N_Def", "P_Def"]

TIPBURN_PATH = os.getenv("TIPBURN_PATH", "artifacts/tipburn_best.pt")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "artifacts/cls_effnetv2_b1_best.pt")

device = "cpu"

tipburn_model = YOLO(TIPBURN_PATH)

def load_classifier(pt_path: str):
    try:
        m = torch.jit.load(pt_path, map_location=device)
        m.eval()
        return m, "torchscript"
    except Exception:
        pass

    model = timm.create_model("tf_efficientnetv2_b1", pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(pt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        try:
            ckpt.eval()
            return ckpt, "full_model"
        except Exception:
            raise RuntimeError("Classifier .pt format not recognized. Re-save as state_dict or TorchScript.")

    fixed = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(fixed, strict=False)

    if len(missing) > 50 or len(unexpected) > 50:
        raise RuntimeError(
            f"Bad weight load (likely wrong timm model name). missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.eval()
    return model, "timm_state_dict"

classifier, classifier_mode = load_classifier(CLASSIFIER_PATH)

cfg = resolve_model_data_config(classifier)
preprocess = create_transform(**cfg, is_training=False)

def softmax_np(x: np.ndarray):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

def predict_from_image_bytes(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    image_area = float(w * h)

    # ---- Classifier ----
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        out = classifier(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.squeeze(0).detach().cpu().numpy().astype(float)

    if (out.min() >= 0.0) and (out.max() <= 1.0) and (abs(out.sum() - 1.0) < 0.05):
        probs_arr = out
    else:
        probs_arr = softmax_np(out)

    probs = {CLASSES[i]: float(probs_arr[i]) for i in range(len(CLASSES))}

    # ---- Tipburn (YOLO) ----
    res = tipburn_model(img, verbose=False)[0]
    boxes = res.boxes

    if boxes is None or len(boxes) == 0:
        num_boxes = 0
        A = 0.0
        C = 0.0
    else:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        A = float(areas.sum() / image_area)
        C = float(conf.mean())
        num_boxes = int(len(conf))

    tip = {
        "num_boxes": num_boxes,
        "A": max(0.0, min(1.0, A)),
        "C": max(0.0, min(1.0, C)),
    }

    hs = compute_health_score(probs, tip)

    return {
        "classifier_mode": classifier_mode,
        "probs": probs,
        "tipburn": tip,
        **hs,
    }

def predict_annotated_image_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    y = tipburn_model(img, verbose=False)[0]
    annotated = draw_tipburn_boxes(img, y)
    return pil_to_png_bytes(annotated)