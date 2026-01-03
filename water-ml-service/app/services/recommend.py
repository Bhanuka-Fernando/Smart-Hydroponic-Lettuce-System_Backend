def recommend(pH: float, EC: float, temp: float, DO: float, risk_level: str, anomaly_flag: int):
    recs = []

    # Prioritize DO issues
    if DO < 6.0:
        recs.append({
            "priority": "HIGH",
            "issue": "Low Dissolved Oxygen (DO)",
            "actions": [
                "Check aeration pump power and airflow",
                "Inspect air stones for blockage",
                "Increase circulation and recheck DO in 10 minutes",
            ],
            "reason": "DO is below the safe threshold; low DO can stress roots quickly."
        })

    if pH < 5.5 or pH > 6.5:
        recs.append({
            "priority": "MED",
            "issue": "pH out of optimal range",
            "actions": [
                "Adjust pH slowly using pH-up / pH-down solution",
                "Retest after 10–20 minutes",
                "Avoid large sudden corrections",
            ],
            "reason": "pH drift affects nutrient uptake and can reduce growth."
        })

    if EC < 1.0:
        recs.append({
            "priority": "MED",
            "issue": "EC too low (nutrients low)",
            "actions": [
                "Check nutrient dosing schedule",
                "Increase nutrient concentration gradually",
                "Confirm EC sensor calibration",
            ],
            "reason": "Low EC may indicate insufficient nutrient concentration."
        })
    elif EC > 2.0:
        recs.append({
            "priority": "MED",
            "issue": "EC too high (nutrients concentrated)",
            "actions": [
                "Dilute by adding clean water gradually",
                "Check dosing equipment for overfeeding",
                "Recheck EC after 10 minutes",
            ],
            "reason": "High EC can cause osmotic stress and reduce water uptake."
        })

    if temp < 18.0 or temp > 26.0:
        recs.append({
            "priority": "LOW",
            "issue": "Water temperature outside ideal range",
            "actions": [
                "Check chiller/heater operation",
                "Insulate tank if needed",
                "Monitor temperature trend for next hour",
            ],
            "reason": "Temperature affects dissolved oxygen and plant metabolism."
        })

    if anomaly_flag == 1:
        recs.insert(0, {
            "priority": "HIGH",
            "issue": "Anomalous water behavior detected",
            "actions": [
                "Check pumps/aeration and sensor connections",
                "Inspect for sudden changes (leaks, blockage, dosing)",
                "Verify readings with a manual check if possible",
            ],
            "reason": "Anomaly detection flagged an unusual pattern compared to normal behavior."
        })

    # If nothing triggered
    if not recs:
        recs.append({
            "priority": "LOW",
            "issue": "Stable conditions",
            "actions": ["Continue monitoring", "Next check in 10 minutes"],
            "reason": "All parameters are within safe ranges and no anomaly detected."
        })

    return recs[:3]
