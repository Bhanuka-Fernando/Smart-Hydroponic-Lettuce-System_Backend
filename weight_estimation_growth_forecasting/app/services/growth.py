# app/services/growth.py
import math
import pickle
from functools import lru_cache
from typing import Tuple, Optional, List, Any

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

# Default column set (ONLY used if we cannot detect cols from bundle/model)
DEFAULT_FEATURE_COLS = [
    "DAP",
    "A_t_cm2",
    "D_t_cm",
    "deltaA_cm2",
    "RGR",
    "airT_mean_3d_C",
    "RH_mean_3d_pct",
    "EC_mean_3d_mScm",
    "pH_mean_3d",
]


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
def _model():
    b = _bundle()

    # your notebook saves it as "model2"
    if isinstance(b, dict) and "model2" in b and hasattr(b["model2"], "predict"):
        return b["model2"]

    # fallback: scan dict values
    if isinstance(b, dict):
        for v in b.values():
            if hasattr(v, "predict"):
                return v

    if hasattr(b, "predict"):
        return b

    raise RuntimeError("Growth bundle loaded but no predictor with .predict() found.")


@lru_cache(maxsize=1)
def _feature_cols() -> List[str]:
    """
    Priority:
      1) bundle['feature_cols']
      2) sklearn model.feature_names_in_
      3) DEFAULT_FEATURE_COLS
    """
    b = _bundle()
    if isinstance(b, dict):
        cols = b.get("feature_cols") or b.get("features") or b.get("columns")
        if cols and isinstance(cols, (list, tuple)):
            return list(cols)

    m = _model()
    if hasattr(m, "feature_names_in_"):
        try:
            return list(getattr(m, "feature_names_in_"))
        except Exception:
            pass

    # last resort
    return list(DEFAULT_FEATURE_COLS)


def compute_deltaA_RGR(A_t_cm2: float, A_prev_cm2: float) -> Tuple[float, float]:
    deltaA = float(A_t_cm2) - float(A_prev_cm2)
    rgr = float(math.log((float(A_t_cm2) + EPS) / (float(A_prev_cm2) + EPS)))
    return deltaA, rgr


def _make_X(row: dict, feat_cols: List[str]):
    if pd is not None:
        return pd.DataFrame([row], columns=feat_cols)
    return np.array([[row[c] for c in feat_cols]], dtype=float)


def _get_sensor(sensors: Any, name: str, fallback: float = 0.0) -> float:
    """
    Works with:
      - sensors.airT_mean_3d_C (your DB means object)
      - sensors.airT / RH / EC / pH (raw)
      - dict style
    """
    if sensors is None:
        return float(fallback)

    # dict
    if isinstance(sensors, dict):
        v = sensors.get(name)
        if v is None:
            # try raw names mapping
            raw_map = {
                "airT_mean_3d_C": "airT",
                "RH_mean_3d_pct": "RH",
                "EC_mean_3d_mScm": "EC",
                "pH_mean_3d": "pH",
            }
            v = sensors.get(raw_map.get(name, ""))
        return float(v) if v is not None else float(fallback)

    # object attrs
    v = getattr(sensors, name, None)
    if v is None:
        raw_map = {
            "airT_mean_3d_C": "airT",
            "RH_mean_3d_pct": "RH",
            "EC_mean_3d_mScm": "EC",
            "pH_mean_3d": "pH",
        }
        v = getattr(sensors, raw_map.get(name, ""), None)

    return float(v) if v is not None else float(fallback)


def predict_tomorrow(
    dap: int,
    A_t_cm2: float,
    D_t_cm: float,
    A_prev_cm2: float,
    sensors,
) -> Tuple[float, float]:
    feat_cols = _feature_cols()

    deltaA_cm2, RGR = compute_deltaA_RGR(A_t_cm2, A_prev_cm2)

    row = {
        "DAP": float(dap),
        "A_t_cm2": float(A_t_cm2),
        "D_t_cm": float(D_t_cm),
        "deltaA_cm2": float(deltaA_cm2),
        "RGR": float(RGR),
        "airT_mean_3d_C": _get_sensor(sensors, "airT_mean_3d_C", 0.0),
        "RH_mean_3d_pct": _get_sensor(sensors, "RH_mean_3d_pct", 0.0),
        "EC_mean_3d_mScm": _get_sensor(sensors, "EC_mean_3d_mScm", 0.0),
        "pH_mean_3d": _get_sensor(sensors, "pH_mean_3d", 0.0),
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
    model = _model()

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
            "airT_mean_3d_C": _get_sensor(sensors, "airT_mean_3d_C", 0.0),
            "RH_mean_3d_pct": _get_sensor(sensors, "RH_mean_3d_pct", 0.0),
            "EC_mean_3d_mScm": _get_sensor(sensors, "EC_mean_3d_mScm", 0.0),
            "pH_mean_3d": _get_sensor(sensors, "pH_mean_3d", 0.0),
        }

        X = _make_X(row, feat_cols)
        A_next, D_next = model.predict(X)[0]
        A_next, D_next = float(A_next), float(D_next)

        out.append(
            {"step": step, "DAP_pred": dap + 1, "A_pred_cm2": A_next, "D_pred_cm": D_next}
        )

        A_prev = A_curr
        A_curr = A_next
        D_curr = D_next
        dap += 1

    return out