def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def compute_health_score(probs: dict, tip: dict):
    # Classifier probs
    p_healthy = float(probs.get("Healthy", 0.0))

    # Step 1) Class risk
    risk_cls = 1.0 - p_healthy

    # Step 2) Tipburn risk
    A = float(tip.get("A", 0.0))          
    C = float(tip.get("C", 0.0))          
    num_boxes = int(tip.get("num_boxes", 0))

    if num_boxes == 0:
        A_cap = 0.0
        risk_tip = 0.0
    else:
        A_cap = min(A / 0.10, 1.0)        
        risk_tip = A_cap * C              

    # Step 3) Combined risk -> health
    risk_total = 0.70 * risk_cls + 0.30 * risk_tip
    health = round(100 * (1 - risk_total))
    health = int(clamp(health, 0, 100))

    # Step 4) Alerts
    if health >= 80:
        status = "OK"
    elif health >= 60:
        status = "WATCH"
    else:
        status = "ACT NOW"

    # Step 5) Main issue
    keys = ["Bacterial", "Fungal", "N_Def", "P_Def", "K_Def", "Healthy"]
    main_issue = max(keys, key=lambda k: float(probs.get(k, 0.0)))

    tipburn_present = (num_boxes > 0 and C >= 0.25)

    return {
        "health_score": health,
        "status": status,
        "main_issue": main_issue,
        "risk_cls": float(clamp(risk_cls, 0, 1)),
        "risk_tip": float(clamp(risk_tip, 0, 1)),
        "risk_total": float(clamp(risk_total, 0, 1)),
        "tipburn_present": bool(tipburn_present),
        "tipburn_A": float(clamp(A, 0, 1)),
        "tipburn_C": float(clamp(C, 0, 1)),
        "tipburn_A_cap": float(clamp(A_cap, 0, 1)),
    }
