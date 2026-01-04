from typing import Optional, Dict, List

def compute_health_score(ph: float, ec_or_tds: Optional[float], temp_c: float, turb_ntu: float, mode: str="EC") -> Dict:
    reasons: List[str] = []
    actions: List[str] = []
    score = 100.0

    # pH
    if ph < 5.5 or ph > 7.5:
        score -= 35; reasons.append("pH out of critical range")
    elif ph < 6.0 or ph > 7.0:
        score -= 15; reasons.append("pH out of optimal range")

    # Temperature
    if temp_c < 16 or temp_c > 28:
        score -= 30; reasons.append("Temperature out of critical range")
    elif temp_c < 18 or temp_c > 24:
        score -= 12; reasons.append("Temperature out of optimal range")

    # Turbidity
    if turb_ntu > 120:
        score -= 30; reasons.append("Turbidity very high")
    elif turb_ntu > 60:
        score -= 15; reasons.append("Turbidity high")

    # EC/TDS optional
    if ec_or_tds is not None:
        if mode.upper() == "EC":
            if ec_or_tds < 0.8 or ec_or_tds > 2.5:
                score -= 10; reasons.append("EC out of typical range")
        else:
            if ec_or_tds < 500 or ec_or_tds > 1500:
                score -= 10; reasons.append("TDS out of typical range")

    score = max(0.0, min(100.0, score))

    if score < 60:
        score_status = "CRITICAL"
        actions = ["Check pH/EC immediately", "Inspect turbidity/filters", "Do partial water change if needed"]
    elif score < 80:
        score_status = "WARNING"
        actions = ["Monitor closely", "Recheck sensors in 10–15 minutes", "Prepare corrective action"]
    else:
        score_status = "OK"
        actions = ["Continue monitoring"]

    return {
        "health_score": score,
        "score_status": score_status,
        "reasons": reasons,
        "actions": actions
    }
