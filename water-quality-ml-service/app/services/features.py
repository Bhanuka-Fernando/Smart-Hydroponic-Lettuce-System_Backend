from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

SENSOR_COLS = ["ph", "temp_c", "turb_ntu", "ec"]
HistoryRow = Tuple[str, float, float, float, float]  # (timestamp_iso, ph, temp_c, turb_ntu, ec)

def build_feature_frame(rows: List[HistoryRow], resample_rule: str) -> pd.DataFrame:
    """
    Convert raw rows to a resampled time-series DF + engineered features.
    """
    df = pd.DataFrame(rows, columns=["timestamp", "ph", "temp_c", "turb_ntu", "ec"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for c in SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp"] + SENSOR_COLS).sort_values("timestamp")
    if df.empty:
        return df

    # Resample to fixed interval
    df = df.set_index("timestamp").resample(resample_rule).mean().dropna().reset_index()

    # Rolling + delta features
    for c in SENSOR_COLS:
        df[f"{c}_rm2"] = df[c].rolling(2, min_periods=2).mean()
        df[f"{c}_rm4"] = df[c].rolling(4, min_periods=4).mean()
        df[f"{c}_rs2"] = df[c].rolling(2, min_periods=2).std()
        df[f"{c}_d2"]  = df[c].diff(2)

    return df

def latest_feature_row(df_feat: pd.DataFrame, feature_cols: List[str]) -> Optional[np.ndarray]:
    if df_feat is None or df_feat.empty:
        return None
    last = df_feat.iloc[-1]
    # Must have all features (rolling windows ready)
    if last[feature_cols].isna().any():
        return None
    return last[feature_cols].to_numpy(dtype=float)

def get_turb_delta_30min(df_feat: pd.DataFrame) -> Optional[float]:
    if df_feat is None or df_feat.empty:
        return None
    v = df_feat.iloc[-1].get("turb_ntu_d2")
    if v is None:
        return None
    try:
        if np.isnan(float(v)):
            return None
        return float(v)
    except Exception:
        return None