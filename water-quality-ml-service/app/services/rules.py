from __future__ import annotations

from typing import List, Optional, Tuple

def water_rule_checks(ph: float, temp_c: float, turb: float, ec: float, turb_d2: Optional[float]) -> Tuple[str, int, List[str], List[str]]:
    """
    Returns: rule_status, severity(0/1/2), reasons, actions
    Starter thresholds (tune for your farm).
    """
    reasons: List[str] = []
    actions: set[str] = set()

    # pH
    if ph < 5.2 or ph > 6.8:
        reasons.append("pH out of critical range")
        actions.update({"Check pH sensor calibration", "Correct pH and recheck after 15 minutes"})
        sev_ph = 2
    elif ph < 5.5 or ph > 6.5:
        reasons.append("pH out of typical range")
        actions.update({"Monitor pH closely", "Recheck after 15 minutes"})
        sev_ph = 1
    else:
        sev_ph = 0

    # Temperature
    if temp_c < 15 or temp_c > 28:
        reasons.append("Temperature out of critical range")
        actions.update({"Check chiller/heater", "Improve water circulation"})
        sev_t = 2
    elif temp_c < 18 or temp_c > 24:
        reasons.append("Temperature out of typical range")
        actions.update({"Monitor temperature", "Improve circulation if needed"})
        sev_t = 1
    else:
        sev_t = 0

    # Turbidity
    if turb > 20:
        reasons.append("Turbidity very high")
        actions.update({"Inspect filters and water flow", "Check for contamination/biofilm", "Consider partial water change"})
        sev_tb = 2
    elif turb > 5:
        reasons.append("Turbidity high")
        actions.update({"Inspect filters and water flow", "Monitor turbidity trend"})
        sev_tb = 1
    else:
        sev_tb = 0

    # Turbidity trend
    if turb_d2 is not None:
        if turb_d2 >= 10:
            reasons.append("Turbidity rising quickly")
            actions.update({"Reduce light hitting the water surface", "Recheck readings in 15 minutes"})
            sev_tr = 1
        elif turb_d2 >= 5:
            reasons.append("Turbidity rising")
            actions.update({"Reduce light hitting the water surface"})
            sev_tr = 1
        else:
            sev_tr = 0
    else:
        sev_tr = 0

    # EC
    if ec < 0.9 or ec > 2.2:
        reasons.append("EC out of critical range")
        actions.update({"Check nutrient dosing system", "Correct EC and recheck after 15 minutes"})
        sev_ec = 2
    elif ec < 1.2 or ec > 1.8:
        reasons.append("EC out of typical range")
        actions.update({"Monitor EC", "Inspect dosing/pumps"})
        sev_ec = 1
    else:
        sev_ec = 0

    sev = max(sev_ph, sev_t, sev_tb, sev_tr, sev_ec)
    rule_status = ["OK", "WARNING", "CRITICAL"][sev]
    return rule_status, sev, reasons, sorted(actions)

def algae_reasoning(turb: float, turb_d2: Optional[float], temp_c: float, ec: float, ph: float) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    actions: set[str] = set()

    if turb > 20:
        reasons.append("Water clarity is poor (turbidity very high)")
        actions.update({"Inspect filters and clean tank walls", "Consider partial water change"})
    elif turb > 5:
        reasons.append("Water clarity is deteriorating (turbidity elevated)")
        actions.update({"Inspect filters and clean tank walls"})

    if turb_d2 is not None and turb_d2 >= 5:
        reasons.append("Turbidity trend indicates increasing algae risk")
        actions.update({"Reduce light exposure to water surface", "Recheck readings in 15 minutes"})

    if temp_c > 24:
        reasons.append("Water temperature is high, which can accelerate algae growth")
        actions.update({"Improve cooling/shading if possible", "Increase circulation"})

    if ec > 1.8:
        reasons.append("Nutrient concentration is high, which can support algae growth")
        actions.update({"Check nutrient dosing and EC calibration"})

    if ph > 6.6:
        reasons.append("pH is elevated; monitor for algae/biofilm activity")
        actions.update({"Inspect channels/tank walls for biofilm"})

    return reasons, sorted(actions)

def health_score_from_severity(sev: int, reasons: List[str]) -> int:
    # Stable demo-safe score
    score = 100 - (sev * 30) - (min(len(reasons), 6) * 5)
    return max(0, min(100, int(score)))