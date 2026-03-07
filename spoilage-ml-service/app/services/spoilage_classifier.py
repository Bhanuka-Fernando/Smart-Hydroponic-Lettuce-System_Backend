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

    def _green_mask(self, arr: np.ndarray) -> np.ndarray:
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)

        mask = (
            (g > 50) &
            (g > r * 1.10) &
            (g > b * 1.08) &
            ((g - r) > 10) &
            ((g - b) > 8)
        )
        return mask

    def _leafy_stats(self, img: Image.Image) -> dict:
        arr = np.array(img.resize((256, 256)), dtype=np.uint8)

        green_mask = self._green_mask(arr)
        global_green_ratio = float(np.mean(green_mask))
        global_green_pixels = int(np.sum(green_mask))

        h, w = arr.shape[:2]
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        center = arr[y1:y2, x1:x2]
        center_mask = self._green_mask(center)
        center_green_ratio = float(np.mean(center_mask))
        center_green_pixels = int(np.sum(center_mask))

        brightness = arr.mean(axis=2)
        dark_ratio = float(np.mean(brightness < 25))
        bright_ratio = float(np.mean(brightness > 245))

        return {
            "global_green_ratio": global_green_ratio,
            "center_green_ratio": center_green_ratio,
            "global_green_pixels": global_green_pixels,
            "center_green_pixels": center_green_pixels,
            "dark_ratio": dark_ratio,
            "bright_ratio": bright_ratio,
        }

    def _validate_basic_image(self, stats: dict):
        if stats["dark_ratio"] > 0.75:
            raise ValueError(
                "Image is too dark. Please capture a clearer top-view lettuce image."
            )

        if stats["bright_ratio"] > 0.75:
            raise ValueError(
                "Image is too bright. Please capture a clearer top-view lettuce image."
            )

    def predict(self, image_bytes: bytes, temperature: float, humidity: float) -> tuple[str, dict]:
        img = self._read_image(image_bytes)

        stats = self._leafy_stats(img)
        self._validate_basic_image(stats)

        print("VALIDATION stats:", stats)

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

        probs_dict = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        sorted_idx = np.argsort(probs)[::-1]
        top1_idx = int(sorted_idx[0])
        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else int(sorted_idx[0])

        top1_conf = float(probs[top1_idx])
        top2_conf = float(probs[top2_idx])
        margin = top1_conf - top2_conf
        stage = self.class_names[top1_idx]

        global_green = stats["global_green_ratio"]
        center_green = stats["center_green_ratio"]
        global_pixels = stats["global_green_pixels"]
        center_pixels = stats["center_green_pixels"]

        weak_visual_evidence = (
            global_green < 0.05 or
            center_green < 0.08 or
            global_pixels < 1800 or
            center_pixels < 500
        )

        very_weak_visual_evidence = (
            global_green < 0.03 or
            center_green < 0.05 or
            global_pixels < 1000 or
            center_pixels < 250
        )

        low_confidence_prediction = (
            top1_conf < 0.72 or
            margin < 0.20
        )

        suspicious_non_lettuce = very_weak_visual_evidence
        uncertain_lettuce = weak_visual_evidence or low_confidence_prediction

        print("SPOILAGE probs:", probs_dict)
        print("SPOILAGE top1_conf:", top1_conf)
        print("SPOILAGE top2_conf:", top2_conf)
        print("SPOILAGE margin:", margin)
        print("SPOILAGE stage:", stage)
        print("SPOILAGE weak_visual_evidence:", weak_visual_evidence)
        print("SPOILAGE very_weak_visual_evidence:", very_weak_visual_evidence)
        print("SPOILAGE low_confidence_prediction:", low_confidence_prediction)

        if suspicious_non_lettuce:
            raise ValueError(
                "Object not recognized as lettuce. Please capture a clear top-view image of one lettuce plant only."
            )

        if uncertain_lettuce:
            raise ValueError(
                "Prediction is uncertain. Please recapture a clear top-view image of the lettuce."
            )

        return stage, probs_dict