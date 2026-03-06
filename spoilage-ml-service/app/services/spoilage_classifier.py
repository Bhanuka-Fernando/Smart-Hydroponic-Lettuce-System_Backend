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
        self.sensor_std = np.array(self.meta["sensor_std"], dtype=np.float32)

        self.model = load_keras_model(model_path)

        # keep these relaxed
        self.min_confidence = 0.45
        self.min_green_ratio = 0.04

    def _read_image(self, image_bytes: bytes) -> Image.Image:
        try:
            img = Image.open(BytesIO(image_bytes))
            img = img.convert("RGB")
            return img
        except UnidentifiedImageError as e:
            raise ValueError("Uploaded file is not a valid image") from e
        except Exception as e:
            print("Image read failed:", repr(e))
            raise ValueError("Uploaded file is not a valid image") from e

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        resized = img.resize((self.img_w, self.img_h))
        x = np.array(resized, dtype=np.float32) / 255.0
        return np.expand_dims(x, axis=0)

    def _preprocess_sensor(self, temperature: float, humidity: float) -> np.ndarray:
        s = np.array([[float(temperature), float(humidity)]], dtype=np.float32)
        s = (s - self.sensor_mean) / (self.sensor_std + 1e-9)
        return s

    def _estimate_green_ratio(self, img: Image.Image) -> float:
        arr = np.array(img.resize((256, 256)), dtype=np.uint8)
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)

        green_mask = (
            (g > 55) &
            (g > r * 1.03) &
            (g > b * 1.02)
        )

        green_ratio = float(np.mean(green_mask))
        return green_ratio

    def _validate_image_content(self, img: Image.Image):
        green_ratio = self._estimate_green_ratio(img)

        print("VALIDATION green_ratio:", green_ratio)

        if green_ratio < self.min_green_ratio:
            raise ValueError(
                "Invalid image. Please capture a clear top-view lettuce image only."
            )

    def predict(self, image_bytes: bytes, temperature: float, humidity: float) -> tuple[str, dict]:
        img = self._read_image(image_bytes)

        # only reject obvious non-lettuce images
        self._validate_image_content(img)

        x_img = self._preprocess_image(img)
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

        max_conf = float(np.max(probs))
        probs_dict = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        stage = max(probs_dict, key=probs_dict.get)

        print("SPOILAGE probs:", probs_dict)
        print("SPOILAGE max_conf:", max_conf)
        print("SPOILAGE stage:", stage)

        # do not reject normal lettuce images too aggressively
        if max_conf < self.min_confidence:
            print("Low confidence, but allowing prediction:", max_conf)

        return stage, probs_dict