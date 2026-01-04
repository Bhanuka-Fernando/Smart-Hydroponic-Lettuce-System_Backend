import math
import pickle
from functools import lru_cache
from typing import Tuple, Optional, List

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import joblib
except Exception:
    joblib = None

from app.core.paths import GROWTH_BUNDLE

EPS = 1e-6


def _load_bundle(path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _bundle():
    return _load_bundle(GROWTH_BUNDLE)


@lru_cache(maxsize=1)
def _feature_cols() -> Optional[List[str]]:
    b = _bundle()
    if isinstance(b, dict) and "feature_cols" in b:
        return list(b["feature_cols"])
    return None


@lru_cache(maxsize=1)
def _model():
    b = _bundle()

    # your notebook saves it as "model2"
    if isinstance(b, dict) and "model2" in b:
        m = b["model2"]
        if hasattr(m, "predict"):
            return m

    # fallback: scan dict values
    if isinstance(b, dict):
        for v in b.values():
            if hasattr(v, "predict"):
                return v

    # bundle itself might be a model
    if hasattr(b, "predict"):
        return b

    raise RuntimeError("Growth bundle loaded but no predictor with .predict() found.")


def compute_deltaA_RGR(A_t_cm2: float, A_prev_cm2: float) -> Tuple[float, float]:
    deltaA = float(A_t_cm2) - float(A_prev_cm2)
    rgr = float(math.log((float(A_t_cm2) + EPS) / (float(A_prev_cm2) + EPS)))
    return deltaA, rgr


def _make_X(row: dict, feat_cols: List[str]):
    """
    Build model input with correct feature names.
    Uses DataFrame if pandas exists; otherwise uses ordered NumPy array.
    """
    if pd is not None:
        return pd.DataFrame([row], columns=feat_cols)
    return np.array([[row[c] for c in feat_cols]], dtype=float)


def predict_tomorrow(
    dap: int,
    A_t_cm2: float,
    D_t_cm: float,
    A_prev_cm2: float,
    sensors,
) -> Tuple[float, float]:
    feat_cols = _feature_cols()
    if not feat_cols:
        raise RuntimeError("Growth bundle missing feature_cols. Re-save bundle with feature_cols.")

    deltaA_cm2, RGR = compute_deltaA_RGR(A_t_cm2, A_prev_cm2)

    row = {
        "DAP": float(dap),
        "A_t_cm2": float(A_t_cm2),
        "D_t_cm": float(D_t_cm),
        "deltaA_cm2": float(deltaA_cm2),
        "RGR": float(RGR),
        "airT_mean_3d_C": float(sensors.airT_mean_3d_C),
        "RH_mean_3d_pct": float(sensors.RH_mean_3d_pct),
        "EC_mean_3d_mScm": float(sensors.EC_mean_3d_mScm),
        "pH_mean_3d": float(sensors.pH_mean_3d),
    }

    X = _make_X(row, feat_cols)
    y = _model().predict(X)[0]  # [A_next, D_next]
    return float(y[0]), float(y[1])


def forecast_n_days(
    dap_start: int,
    A_prev_cm2: float,
    A_t_cm2: float,
    D_t_cm: float,
    sensors,
    n_days: int,
):
    feat_cols = _feature_cols()
    if not feat_cols:
        raise RuntimeError("Growth bundle missing feature_cols. Re-save bundle with feature_cols.")

    model = _model()  # cache once

    out = []
    A_prev = float(A_prev_cm2)
    A_curr = float(A_t_cm2)
    D_curr = float(D_t_cm)
    dap = int(dap_start)

    for step in range(1, int(n_days) + 1):
        deltaA, RGR = compute_deltaA_RGR(A_curr, A_prev)

        row = {
            "DAP": float(dap),
            "A_t_cm2": float(A_curr),
            "D_t_cm": float(D_curr),
            "deltaA_cm2": float(deltaA),
            "RGR": float(RGR),
            "airT_mean_3d_C": float(sensors.airT_mean_3d_C),
            "RH_mean_3d_pct": float(sensors.RH_mean_3d_pct),
            "EC_mean_3d_mScm": float(sensors.EC_mean_3d_mScm),
            "pH_mean_3d": float(sensors.pH_mean_3d),
        }

        X = _make_X(row, feat_cols)
        A_next, D_next = model.predict(X)[0]
        A_next, D_next = float(A_next), float(D_next)

        out.append(
            {
                "step": step,
                "DAP_pred": dap + 1,
                "A_pred_cm2": A_next,
                "D_pred_cm": D_next,
            }
        )

        # shift for next iteration
        A_prev = A_curr
        A_curr = A_next
        D_curr = D_next
        dap += 1

    return out
