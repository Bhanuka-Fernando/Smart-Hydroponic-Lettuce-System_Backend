from __future__ import annotations

import json
from io import BytesIO
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf


def load_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # convert to numpy for fast ops
    meta["sensor_mean"] = np.array(meta["sensor_mean"], dtype=np.float32)  # [mean_temp, mean_hum]
    meta["sensor_std"] = np.array(meta["sensor_std"], dtype=np.float32)    # [std_temp, std_hum]
    meta["img_size"] = tuple(meta["img_size"])                             # (224,224)
    return meta


def load_spoilage_model(model_path: str):
    # loads once at startup
    return tf.keras.models.load_model(model_path)


def _preprocess_image(image_bytes: bytes, img_size: Tuple[int, int]) -> np.ndarray:
    # Colab: tf.cast(img,float32) keep 0..255 then mobilenet_v2.preprocess_input
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32)  # 0..255
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)  # -> [-1,1]
    return np.expand_dims(arr, axis=0)  # (1,H,W,3)


def _preprocess_sensors(temp_c: float, humidity_pct: float, meta: dict) -> np.ndarray:
    x = np.array([temp_c, humidity_pct], dtype=np.float32)  # IMPORTANT: [temp, hum]
    x = (x - meta["sensor_mean"]) / meta["sensor_std"]
    return np.expand_dims(x, axis=0)  # (1,2)


def _safe_probs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    # if already probs, keep
    if y.min() >= -1e-4 and y.max() <= 1.0 + 1e-3 and abs(float(y.sum()) - 1.0) < 1e-2:
        return y
    # else softmax
    y = y - np.max(y)
    exp = np.exp(y)
    return exp / np.sum(exp)


def predict_spoilage(
    model,
    meta: dict,
    image_bytes: bytes,
    temp_c: Optional[float],
    humidity_pct: Optional[float],
) -> Tuple[str, float, Dict[str, float]]:
    x_img = _preprocess_image(image_bytes, meta["img_size"])
    class_names = meta["class_names"]

    # Your model is IMAGE+SENSOR, so it should be 2 inputs.
    if len(model.inputs) == 1:
        y = model.predict(x_img, verbose=0)[0]
    else:
        if temp_c is None or humidity_pct is None:
            raise ValueError("temp_c and humidity_pct are required for IMAGE+SENSOR model.")
        x_s = _preprocess_sensors(temp_c, humidity_pct, meta)
        y = model.predict([x_img, x_s], verbose=0)[0]

    probs = _safe_probs(y)
    idx = int(np.argmax(probs))
    stage = class_names[idx]
    conf = float(probs[idx])

    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return stage, conf, probs_dict
