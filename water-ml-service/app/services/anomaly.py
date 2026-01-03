# app/services/anomaly.py
import json
import os
from collections import deque
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.paths import ARTIFACT_DIR
from app.services.features import build_features

# In-memory prediction history per tank (persistence across requests)
_TANK_HISTORY: Dict[str, deque] = {}


def _load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "isoforest.joblib")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    feature_list = json.loads((ARTIFACT_DIR / "feature_list.json").read_text())

    meta_path = ARTIFACT_DIR / "train_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    return model, scaler, feature_list, meta


MODEL, SCALER, FEATURE_LIST, META = _load_artifacts()


def get_threshold(default: float = 0.65) -> float:
    # env override
    if os.environ.get("THRESHOLD"):
        return float(os.environ["THRESHOLD"])
    # meta override
    if "threshold" in META:
        return float(META["threshold"])
    return default


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _decision_to_anom01(decision: float, k: float = 5.0) -> float:
    """
    IsolationForest decision_function:
      - higher => more normal
      - lower/negative => more abnormal

    Convert to 0..1 anomaly score:
      anom = sigmoid(-k * decision)

    k controls steepness. 5.0 works well for demos.
    """
    return float(np.clip(_sigmoid(-k * decision), 0.0, 1.0))


def persistence_flag_across_requests(tank_id: str, base_pred: int, mode: str = "2of3") -> int:
    """
    Persistence across API calls:
      - "2of3": need 2 positives in last 3
      - "NofN": need N positives in last N (PERSIST_N)
    """
    hist = _TANK_HISTORY.setdefault(tank_id, deque(maxlen=3))
    hist.append(int(base_pred))

    if mode == "NofN":
        n = int(os.environ.get("PERSIST_N", "2"))
        if len(hist) < n:
            return 0
        return 1 if sum(list(hist)[-n:]) >= n else 0

    # default: 2-of-3
    if len(hist) < 3:
        return 0
    return 1 if sum(hist) >= 2 else 0


def persistence_flag_in_window(base_preds: np.ndarray) -> int:
    """
    Persistence inside ONE request window (last 3 points):
      return 1 if >=2 positives in last 3
    """
    base_preds = np.asarray(base_preds, dtype=int)
    if len(base_preds) < 3:
        return 0
    return 1 if int(base_preds[-3:].sum()) >= 2 else 0


def infer_anomaly(tank_id: str, readings: list[dict]) -> Tuple[float, float, int]:
    """
    Returns: (anom_score_0_1, threshold, anomaly_flag)
    """
    if not readings:
        return 0.0, get_threshold(), 0

    df = pd.DataFrame(readings)
    df["tank_id"] = tank_id
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["tank_id", "timestamp"])

    # Create features (must match feature_list.json names)
    df_feat = build_features(df, w=3)

    # Make sure all required features exist to avoid KeyError
    for f in FEATURE_LIST:
        if f not in df_feat.columns:
            df_feat[f] = np.nan

    Xall = df_feat[FEATURE_LIST].astype(float)
    Xvalid = Xall.dropna()

    # Need at least 1 valid row to score last point
    if len(Xvalid) < 1:
        return 0.0, get_threshold(), 0

    # Score only the last valid row
    X_last = Xvalid.tail(1)
    Xs = SCALER.transform(X_last.values)

    # decision_function higher => normal
    decision = float(MODEL.decision_function(Xs)[0])

    # Stable 0..1 anomaly score
    k = float(os.environ.get("SIGMOID_K", "5.0"))
    score01 = _decision_to_anom01(decision, k=k)

    th = get_threshold()

    # base prediction
    base_pred = 1 if score01 >= th else 0

    # persistence mode
    persist_in_window = os.environ.get("PERSIST_IN_WINDOW", "0") == "1"
    mode = os.environ.get("PERSIST_MODE", "2of3")

    if persist_in_window:
        # compute base preds for last few valid rows inside window (optional)
        tail_n = min(10, len(Xvalid))
        X_tail = Xvalid.tail(tail_n)
        Xs_tail = SCALER.transform(X_tail.values)
        decisions = MODEL.decision_function(Xs_tail)
        scores = np.array([_decision_to_anom01(float(d), k=k) for d in decisions])
        base_preds = (scores >= th).astype(int)
        flag = persistence_flag_in_window(base_preds)
    else:
        # across requests (needs 3 API calls to confirm 2-of-3)
        flag = persistence_flag_across_requests(tank_id, base_pred, mode=mode)

    return float(score01), float(th), int(flag)
