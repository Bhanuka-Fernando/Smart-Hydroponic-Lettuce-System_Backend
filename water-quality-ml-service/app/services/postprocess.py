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

def worst_status(a: str, b: str) -> str:
    order = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    return a if order[a] >= order[b] else b

def sensor_quality_checks(ph: float, temp_c: float, turb: float, ec: float) -> Tuple[Literal["OK","SUSPECT"], List[str]]:
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