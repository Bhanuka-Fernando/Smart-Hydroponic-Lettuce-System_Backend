from __future__ import annotations
import numpy as np
from .model_loader import load_json, load_joblib

class RemainingDaysRegressor:
    def __init__(self, model_path: str, meta_path: str):
        self.meta = load_json(meta_path)
        self.feature_cols = self.meta["feature_cols"]

        self.temp_mean = float(self.meta["temp_mean"])
        self.temp_std  = float(self.meta["temp_std"])
        self.hum_mean  = float(self.meta["hum_mean"])
        self.hum_std   = float(self.meta["hum_std"])

        self.model = load_joblib(model_path)

    def predict(self, probs: dict, temperature: float, humidity: float) -> float:
        # Keep the exact feature names used in training meta
        p_fresh = probs.get("fresh", 0.0)
        p_slight = probs.get("slightly_aged", 0.0)
        p_near = probs.get("near_spoilage", 0.0)
        p_spoiled = probs.get("spoiled", 0.0)

        t = (float(temperature) - self.temp_mean) / (self.temp_std + 1e-9)
        h = (float(humidity) - self.hum_mean) / (self.hum_std + 1e-9)

        x = np.array([[p_fresh, p_slight, p_near, p_spoiled, t, h]], dtype=np.float32)
        y = float(self.model.predict(x)[0])
        return max(0.0, y)
