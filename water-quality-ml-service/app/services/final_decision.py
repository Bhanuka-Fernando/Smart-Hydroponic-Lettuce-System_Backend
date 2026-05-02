from __future__ import annotations

from typing import Dict, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def expected_severity(probs: Dict[str, float], order: list[str]) -> float:
    """
    Convert probabilities into severity score [0..2] using expected value.
    order:
      ["OK","WARNING","CRITICAL"] or ["LOW","MEDIUM","HIGH"]
    """
    weights = {order[0]: 0.0, order[1]: 1.0, order[2]: 2.0}
    return sum(weights[k] * float(probs.get(k, 0.0)) for k in weights)


def _worst_status(a: str, b: str) -> str:
    """
    Return the more severe of two statuses using OK < WARNING < CRITICAL.
    """
    order = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def final_status_from_combined(
    rule_status: str,
    water_probs: Dict[str, float],
    algae_probs: Dict[str, float],
    health_score: int,
) -> Tuple[str, float]:
    """
    Combine:
      - rule_status (safety override + minimum severity floor)
      - water model probabilities (OK/WARNING/CRITICAL)
      - algae model probabilities (LOW/MEDIUM/HIGH)
      - health_score (0..90)

    Returns:
      final_status: OK/WARNING/CRITICAL
      final_severity_score: float (0..2)
    """

    # 1) Hard safety override
    if rule_status == "CRITICAL":
        return "CRITICAL", 2.0

    # 2) Convert to severity scores (0..2)
    sev_water = expected_severity(water_probs, ["OK", "WARNING", "CRITICAL"])
    sev_algae = expected_severity(algae_probs, ["LOW", "MEDIUM", "HIGH"])
    sev_score = 2.0 * (1.0 - _clamp(float(health_score), 0.0, 90.0) / 90.0)

    # 3) Weighted combine (tune weights if needed)
    sev_final = 0.60 * sev_water + 0.25 * sev_score + 0.15 * sev_algae
    sev_final = _clamp(sev_final, 0.0, 2.0)

    # 4) Map combined severity to label
    if sev_final >= 1.35:
        label = "CRITICAL"
    elif sev_final >= 0.75:
        label = "WARNING"
    else:
        label = "OK"

    # 5) IMPORTANT: never downgrade below rule_status
    # If rules say WARNING, final cannot be OK.
    # If rules say CRITICAL, we returned CRITICAL earlier.
    final_label = _worst_status(label, rule_status)

    # (Optional) if we upgraded due to rule floor, keep score consistent
    if final_label == "WARNING" and label == "OK":
        sev_final = max(sev_final, 0.75)
    if final_label == "CRITICAL" and label != "CRITICAL":
        sev_final = max(sev_final, 1.35)

    return final_label, sev_final