def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def compute_tipburn_risk(tip: dict):
    num_boxes = int(tip.get("num_boxes", 0))
    A = float(tip.get("A", 0.0))
    C = float(tip.get("C", 0.0))

    if num_boxes == 0:
        A_cap = 0.0
        risk_tip = 0.0
    else:
        A_cap = min(A / 0.10, 1.0)   # 10% coverage treated as max
        risk_tip = A_cap * C

    return {
        "num_boxes": num_boxes,
        "A": clamp(A, 0, 1),
        "C": clamp(C, 0, 1),
        "A_cap": clamp(A_cap, 0, 1),
        "risk_tip": clamp(risk_tip, 0, 1),
        "tipburn_present": bool(num_boxes > 0 and C >= 0.25),
    }


def compute_health_score(probs: dict, tip: dict):
    # --- classification summary ---
    cls_label = max(probs.keys(), key=lambda k: float(probs.get(k, 0.0)))
    cls_conf = float(probs.get(cls_label, 0.0))
    p_healthy = float(probs.get("Healthy", 0.0))

    # --- risks ---
    risk_cls = 1.0 - p_healthy
    tip_calc = compute_tipburn_risk(tip)
    risk_tip = tip_calc["risk_tip"]

    # --- score ---
    risk_total = 0.70 * risk_cls + 0.30 * risk_tip
    health = round(100 * (1 - risk_total))
    health = int(clamp(health, 0, 100))

    # --- decide primary issue (farmer-facing) ---
    # ✅ Always surface tipburn if present
    if tip_calc["tipburn_present"]:
        primary_issue = "Tipburn"
        driver = "tipburn"
    elif cls_label != "Healthy" and cls_conf >= 0.45:
        primary_issue = cls_label
        driver = "classifier"
    else:
        primary_issue = "Healthy"
        driver = "classifier"

    # --- status ---
    if health >= 80:
        status = "OK"
    elif health >= 60:
        status = "WATCH"
    else:
        status = "ACT NOW"

    # soften alert when classifier is healthy & tipburn is minor
    if primary_issue == "Tipburn" and tip_calc["A_cap"] < 0.20 and tip_calc["C"] < 0.60:
        status = "WATCH"

    # --- explanation text ---
    top3 = sorted(probs.items(), key=lambda x: -float(x[1]))[:3]
    reason = (
        f"Primary driver: {driver}. "
        f"Top prediction: {cls_label} ({cls_conf:.2f}), Healthy={p_healthy:.2f}. "
        f"Tipburn: present={tip_calc['tipburn_present']}, boxes={tip_calc['num_boxes']}, "
        f"A={tip_calc['A']:.4f}, C={tip_calc['C']:.2f}. "
        f"Score uses risk_cls=1-Healthy and risk_tip=A_cap*C."
    )

    return {
        "health_score": health,
        "status": status,

        "classification_label": cls_label,
        "classification_confidence": cls_conf,
        "primary_issue": primary_issue,
        "decision_driver": driver,
        "reason": reason,

        "risk_cls": float(clamp(risk_cls, 0, 1)),
        "risk_tip": float(clamp(risk_tip, 0, 1)),
        "risk_total": float(clamp(risk_total, 0, 1)),
        "top3_probs": {k: float(v) for k, v in top3},

        "tipburn_present": tip_calc["tipburn_present"],
        "tipburn_A": tip_calc["A"],
        "tipburn_C": tip_calc["C"],
        "tipburn_A_cap": tip_calc["A_cap"],
    }
