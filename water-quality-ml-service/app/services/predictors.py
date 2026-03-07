from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

class DualPredictor:
    def __init__(self, water_model_path: str, algae_model_path: str):
        self.water_art = self._load(Path(water_model_path))
        self.algae_art = self._load(Path(algae_model_path))

        self.water_model = self.water_art["model"]
        self.algae_model = self.algae_art["model"]

        self.feature_cols = self.water_art["feature_cols"]

        # Make sure both artifacts use same feature set
        if self.algae_art.get("feature_cols") != self.feature_cols:
            raise RuntimeError("Feature columns mismatch between water and algae artifacts.")

        self.water_classes = list(self.water_art["classes"])
        self.algae_classes = list(self.algae_art["classes"])

    def _load(self, path: Path) -> dict:
        if not path.exists():
            raise RuntimeError(f"Missing model artifact: {path}")
        art = joblib.load(path)
        if "model" not in art or "feature_cols" not in art or "classes" not in art:
            raise RuntimeError(f"Invalid artifact format: {path}")
        return art

    def predict(self, x_row: np.ndarray) -> Tuple[str, Dict[str, float], str, Dict[str, float]]:
        """
        x_row: shape (n_features,)
        Returns:
          water_pred, water_probs, algae_pred, algae_probs
        """
        X = x_row.reshape(1, -1)

        w_pred = str(self.water_model.predict(X)[0])
        w_proba = self.water_model.predict_proba(X)[0]
        w_probs = {cls: float(p) for cls, p in zip(self.water_classes, w_proba)}

        a_pred = str(self.algae_model.predict(X)[0])
        a_proba = self.algae_model.predict_proba(X)[0]
        a_probs = {cls: float(p) for cls, p in zip(self.algae_classes, a_proba)}

        return w_pred, w_probs, a_pred, a_probs