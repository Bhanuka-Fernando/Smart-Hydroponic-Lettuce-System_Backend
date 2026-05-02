from __future__ import annotations

from typing import Dict, Optional, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _range_risk(
    x: float,
    ideal_lo: float,
    ideal_hi: float,
    warn_lo: float,
    warn_hi: float,
    crit_lo: float,
    crit_hi: float,
) -> float:
    """Risk in [0,1] based on distance from ideal range (continuous)."""
    if ideal_lo <= x <= ideal_hi:
        return 0.0

    # Below ideal
    if x < ideal_lo:
        if x >= warn_lo:
            return (ideal_lo - x) / (ideal_lo - warn_lo + 1e-9) * 0.35
        if x >= crit_lo:
            return 0.35 + (warn_lo - x) / (warn_lo - crit_lo + 1e-9) * 0.45
        return 1.0

    # Above ideal
    if x > ideal_hi:
        if x <= warn_hi:
            return (x - ideal_hi) / (warn_hi - ideal_hi + 1e-9) * 0.35
        if x <= crit_hi:
            return 0.35 + (x - warn_hi) / (crit_hi - warn_hi + 1e-9) * 0.45
        return 1.0

    return 0.0


def _turb_risk(turb_ntu: float) -> float:
    """
    Turbidity risk (tune later using farm feedback).
    default bands:
      ideal:    0–3
      warning:  3–8
      critical: 8–20+
    """
    if turb_ntu <= 3.0:
        return 0.0
    if turb_ntu <= 8.0:
        return (turb_ntu - 3.0) / (8.0 - 3.0) * 0.60
    if turb_ntu <= 20.0:
        return 0.60 + (turb_ntu - 8.0) / (20.0 - 8.0) * 0.35
    return 1.0


def _trend_risk(turb_d2: Optional[float]) -> float:
    """Trend penalty from turbidity delta over ~30 minutes."""
    if turb_d2 is None:
        return 0.0
    if turb_d2 <= 1.0:
        return 0.0
    if turb_d2 <= 5.0:
        return (turb_d2 - 1.0) / (5.0 - 1.0) * 0.5
    return 0.8


def compute_health_score(
    ph: float,
    temp_c: float,
    turb_ntu: float,
    ec: float,
    turb_d2: Optional[float] = None,
    ml_conf: Optional[float] = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Returns:
      score: int in [0..90] (capped at 90 intentionally)
      breakdown: dict for debugging
    """
    r_ph = _range_risk(ph, ideal_lo=5.8, ideal_hi=6.3, warn_lo=5.5, warn_hi=6.6, crit_lo=5.2, crit_hi=6.9)
    r_temp = _range_risk(temp_c, ideal_lo=19.0, ideal_hi=23.0, warn_lo=18.0, warn_hi=24.5, crit_lo=15.0, crit_hi=28.0)
    r_ec = _range_risk(ec, ideal_lo=1.3, ideal_hi=1.7, warn_lo=1.2, warn_hi=1.8, crit_lo=0.9, crit_hi=2.2)
    r_turb = _turb_risk(turb_ntu)
    r_trend = _trend_risk(turb_d2)

    total_risk = 0.20 * r_ph + 0.20 * r_temp + 0.25 * r_ec + 0.35 * r_turb + 0.15 * r_trend
    total_risk = _clamp(total_risk, 0.0, 1.0)

    conf_penalty = 0.0
    if ml_conf is not None:
        conf_penalty = (1.0 - _clamp(float(ml_conf), 0.0, 1.0)) * 10.0  # up to -10 points

    score = int(round(90.0 * (1.0 - total_risk) - conf_penalty))
    score = max(0, min(90, score))

    breakdown = {
        "risk_ph": r_ph,
        "risk_temp": r_temp,
        "risk_ec": r_ec,
        "risk_turb": r_turb,
        "risk_trend": r_trend,
        "risk_total": total_risk,
        "conf_penalty": conf_penalty,
    }

    return score, breakdown