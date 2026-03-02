def make_status(stage: str, probs: dict) -> str:
    p_spoiled = probs.get("spoiled", 0.0)
    p_near = probs.get("near_spoilage", 0.0)

    if stage == "spoiled" or p_spoiled >= 0.60:
        return "❌ SPOILED: Not recommended for sale/consumption."
    if stage == "near_spoilage" or p_near >= 0.60:
        return "🚨 ALERT: Near spoilage detected. Harvest / use soon!"
    if stage == "slightly_aged" and p_near >= 0.30:
        return "⚠️ Early warning: Quality dropping. Plan to use soon."
    return "✅ OK: Fresh / acceptable quality."
