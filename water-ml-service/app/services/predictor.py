import numpy as np
import pandas as pd
from typing import Dict, List

from app.services.model_loader import load_artifact
from app.services.score_rules import compute_health_score


def fuse_status(ml_status: str, p_crit: float, score_status: str) -> str:
    if ml_status == "CRITICAL" and p_crit >= 0.60:
        return "CRITICAL"
    if score_status == "CRITICAL":
        return "CRITICAL"
    if ml_status == "WARNING" or score_status == "WARNING":
        return "WARNING"
    return "OK"


def predict_one(reading: Dict) -> Dict:
    """
    Single reading (no history).
    reading keys: ph, temp_c, turb_ntu, ec_or_tds(optional), mode(optional)
    """
    art = load_artifact()
    model = art["model"]
    feats = art["features"]
    label_map = art["label_map"]

    ph = float(reading["ph"])
    temp_c = float(reading["temp_c"])
    turb_ntu = float(reading["turb_ntu"])
    ec_or_tds = reading.get("ec_or_tds", None)
    mode = reading.get("mode", "EC")

    # minimal feature vector (no history)
    feat_dict = {f: 0.0 for f in feats}
    feat_dict["ph"] = ph
    feat_dict["temp_c"] = temp_c
    feat_dict["turb_ntu"] = turb_ntu

    # approximate roll means with current values, deltas/std = 0
    for k in ["ph_rm2", "ph_rm4"]:
        if k in feat_dict:
            feat_dict[k] = ph
    for k in ["temp_rm2", "temp_rm4"]:
        if k in feat_dict:
            feat_dict[k] = temp_c
    for k in ["turb_rm2", "turb_rm4"]:
        if k in feat_dict:
            feat_dict[k] = turb_ntu

    for k in ["ph_rs2", "temp_rs2", "turb_rs2", "ph_d2", "temp_d2", "turb_d2"]:
        if k in feat_dict:
            feat_dict[k] = 0.0

    X = pd.DataFrame([[float(feat_dict[f]) for f in feats]], columns=feats)
    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    ml_status = label_map[pred]

    score_out = compute_health_score(ph=ph, ec_or_tds=ec_or_tds, temp_c=temp_c, turb_ntu=turb_ntu, mode=mode)
    final_status = fuse_status(ml_status, float(proba[2]), score_out["score_status"])

    return {
        "status_model": ml_status,
        "proba_ok": float(proba[0]),
        "proba_warn": float(proba[1]),
        "proba_crit": float(proba[2]),
        "health_score": float(score_out["health_score"]),
        "score_status": score_out["score_status"],
        "final_status": final_status,
        "reasons": score_out["reasons"],
        "actions": score_out["actions"],
    }


def _build_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("created_at").set_index("created_at")

    # 15-min resample
    df15 = df.resample("15min").mean(numeric_only=True)
    df15 = df15.dropna(subset=["ph", "temp_c", "turb_ntu"], how="any")
    if len(df15) < 4:
        return pd.DataFrame()

    # rolling windows: 2 points=30min, 4 points=60min
    df15["ph_rm2"] = df15["ph"].rolling(2, min_periods=2).mean()
    df15["ph_rs2"] = df15["ph"].rolling(2, min_periods=2).std(ddof=0)
    df15["ph_rm4"] = df15["ph"].rolling(4, min_periods=4).mean()
    df15["ph_d2"] = df15["ph"] - df15["ph"].shift(2)

    df15["temp_rm2"] = df15["temp_c"].rolling(2, min_periods=2).mean()
    df15["temp_rs2"] = df15["temp_c"].rolling(2, min_periods=2).std(ddof=0)
    df15["temp_rm4"] = df15["temp_c"].rolling(4, min_periods=4).mean()
    df15["temp_d2"] = df15["temp_c"] - df15["temp_c"].shift(2)

    df15["turb_rm2"] = df15["turb_ntu"].rolling(2, min_periods=2).mean()
    df15["turb_rs2"] = df15["turb_ntu"].rolling(2, min_periods=2).std(ddof=0)
    df15["turb_rm4"] = df15["turb_ntu"].rolling(4, min_periods=4).mean()
    df15["turb_d2"] = df15["turb_ntu"] - df15["turb_ntu"].shift(2)

    need = ["ph_rm2","temp_rm2","turb_rm2","ph_rs2","temp_rs2","turb_rs2","ph_rm4","temp_rm4","turb_rm4","ph_d2","temp_d2","turb_d2"]
    df15 = df15.dropna(subset=need, how="any")
    if df15.empty:
        return pd.DataFrame()

    return df15.reset_index()


def predict_batch(payload: Dict) -> Dict:
    """
    Batch readings (history). Returns prediction for latest resampled point.
    """
    art = load_artifact()
    model = art["model"]
    feats = art["features"]
    label_map = art["label_map"]

    readings: List[Dict] = payload["readings"]
    pond_id = payload.get("pond_id", "POND_01")

    df = pd.DataFrame([{
        "created_at": r["timestamp"],
        "ph": float(r["ph"]),
        "temp_c": float(r["temp_c"]),
        "turb_ntu": float(r["turb_ntu"]),
    } for r in readings])

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at", "ph", "temp_c", "turb_ntu"])
    df = df.drop_duplicates(subset=["created_at"]).sort_values("created_at")

    if len(df) < 2:
        raise ValueError("Not enough valid readings after cleaning.")

    df_feat = _build_timeseries_features(df)
    if df_feat.empty:
        # fallback to single-reading mode (latest raw)
        last_raw = df.iloc[-1].to_dict()
        return {
            **predict_one({
                "ph": float(last_raw["ph"]),
                "temp_c": float(last_raw["temp_c"]),
                "turb_ntu": float(last_raw["turb_ntu"]),
                "ec_or_tds": readings[-1].get("ec_or_tds", None),
                "mode": readings[-1].get("mode", "EC"),
            }),
            "meta": {
                "pond_id": str(pond_id),
                "mode": "fallback_single_reading",
                "note": "Not enough history for rolling features (need ~1 hour)."
            }
        }

    last = df_feat.iloc[-1]

    X = pd.DataFrame([[float(last[f]) for f in feats]], columns=feats)
    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    ml_status = label_map[pred]

    ph = float(last["ph"])
    temp_c = float(last["temp_c"])
    turb_ntu = float(last["turb_ntu"])
    ec_or_tds = readings[-1].get("ec_or_tds", None)
    mode = readings[-1].get("mode", "EC")

    score_out = compute_health_score(ph=ph, ec_or_tds=ec_or_tds, temp_c=temp_c, turb_ntu=turb_ntu, mode=mode)
    final_status = fuse_status(ml_status, float(proba[2]), score_out["score_status"])

    return {
        "status_model": ml_status,
        "proba_ok": float(proba[0]),
        "proba_warn": float(proba[1]),
        "proba_crit": float(proba[2]),
        "health_score": float(score_out["health_score"]),
        "score_status": score_out["score_status"],
        "final_status": final_status,
        "reasons": score_out["reasons"],
        "actions": score_out["actions"],
        "meta": {
            "pond_id": str(pond_id),
            "mode": "batch_timeseries",
            "resampled_points": str(len(df_feat)),
            "latest_time": str(last["created_at"]),
        }
    }
