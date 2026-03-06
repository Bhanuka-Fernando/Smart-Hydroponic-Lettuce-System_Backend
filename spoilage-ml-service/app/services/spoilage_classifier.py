from __future__ import annotations
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO

from .model_loader import load_json, load_keras_model

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

class SpoilageClassifier:
    def __init__(self, model_path: str, meta_path: str):
        self.meta = load_json(meta_path)
        self.class_names = self.meta["class_names"]
        self.img_w, self.img_h = self.meta["img_size"][0], self.meta["img_size"][1]

        self.sensor_mean = np.array(self.meta["sensor_mean"], dtype=np.float32)
        self.sensor_std  = np.array(self.meta["sensor_std"], dtype=np.float32)

        self.model = load_keras_model(model_path)

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            print("PREPROCESS image bytes length:", len(image_bytes) if image_bytes else 0)
            print("PREPROCESS first 20 bytes:", image_bytes[:20])

            img = Image.open(BytesIO(image_bytes))
            print("PREPROCESS image format:", img.format)
            print("PREPROCESS image mode:", img.mode)
            print("PREPROCESS image size:", img.size)

            img = img.convert("RGB")
            img = img.resize((self.img_w, self.img_h))
            x = np.array(img, dtype=np.float32) / 255.0
            return np.expand_dims(x, axis=0)

        except UnidentifiedImageError as e:
            print("PIL could not identify image file. First 50 bytes:", image_bytes[:50])
            raise ValueError("Uploaded file is not a valid image") from e

        except Exception as e:
            print("Image preprocessing failed:", repr(e))
            raise

    def _preprocess_sensor(self, temperature: float, humidity: float) -> np.ndarray:
        s = np.array([[float(temperature), float(humidity)]], dtype=np.float32)
        s = (s - self.sensor_mean) / (self.sensor_std + 1e-9)
        return s

    def predict(self, image_bytes: bytes, temperature: float, humidity: float) -> tuple[str, dict]:
        x_img = self._preprocess_image(image_bytes)
        x_sens = self._preprocess_sensor(temperature, humidity)

        if isinstance(self.model.inputs, (list, tuple)) and len(self.model.inputs) == 2:
            y = self.model.predict([x_img, x_sens], verbose=0)
        else:
            y = self.model.predict(x_img, verbose=0)

        probs = np.array(y[0], dtype=np.float32).flatten()

        s = float(np.sum(probs))
        if not (0.95 <= s <= 1.05):
            probs = _softmax(probs)
        else:
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / (np.sum(probs) + 1e-9)

        probs_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        stage = max(probs_dict, key=probs_dict.get)
        return stage, probs_dict