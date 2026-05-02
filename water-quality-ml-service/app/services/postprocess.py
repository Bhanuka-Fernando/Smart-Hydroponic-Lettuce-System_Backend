from __future__ import annotations

from typing import Dict, List, Literal, Tuple


def confidence_gate_ml(status: str, probs: Dict[str, float], sensor_quality: str) -> str:
    """
    Avoid ML-only CRITICAL on noise: require prob >= 0.80 and sensor_quality OK.
    """
    if status == "CRITICAL":
        p = float(probs.get("CRITICAL", 0.0))
        if p < 0.80:
            return "WARNING"
        if sensor_quality != "OK":
            return "WARNING"
    return status


def gate_water_probs(ml_status: str, probs: Dict[str, float], sensor_quality: str) -> Dict[str, float]:
    """
    Adjust probabilities to match the same safety gating used for labels.
    If ML predicts CRITICAL with low confidence or suspect sensors, shift CRITICAL prob to WARNING.
    """
    out = dict(probs)

    if ml_status == "CRITICAL":
        pcrit = float(out.get("CRITICAL", 0.0))
        if pcrit < 0.80 or sensor_quality != "OK":
            out["WARNING"] = float(out.get("WARNING", 0.0)) + pcrit
            out["CRITICAL"] = 0.0

    # Renormalize
    s = sum(float(v) for v in out.values())
    if s > 0:
        out = {k: float(v) / s for k, v in out.items()}
    return out


def worst_status(a: str, b: str) -> str:
    order = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    return a if order[a] >= order[b] else b


def sensor_quality_checks(ph: float, temp_c: float, turb: float, ec: float) -> Tuple[Literal["OK", "SUSPECT"], List[str]]:
    notes: List[str] = []

    if not (0 <= ph <= 14):
        notes.append("pH out of physical bounds")
    if temp_c < 0 or temp_c > 60:
        notes.append("Temperature out of physical bounds")
    if ec < 0:
        notes.append("EC negative (invalid)")
    if turb < 0:
        notes.append("Turbidity negative (invalid)")

    return ("SUSPECT" if notes else "OK"), notes