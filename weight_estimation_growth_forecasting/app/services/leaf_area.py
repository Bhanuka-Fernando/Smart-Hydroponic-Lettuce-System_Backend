# ml-service/app/services/leaf_area.py
import json
import math
from functools import lru_cache

from app.core.paths import LEAF_AREA_JSON


@lru_cache(maxsize=1)
def _load_params():
    if not LEAF_AREA_JSON.exists():
        raise FileNotFoundError(f"Missing params file: {LEAF_AREA_JSON}")

    p = json.loads(LEAF_AREA_JSON.read_text())

    intercept = float(p["intercept"])
    coef_log_A = float(p["coef_log_A"])
    coef_log_D = float(p["coef_log_D"])

    # optional: if you add "multiplier" in json, it will be applied
    multiplier = float(p.get("multiplier", 1.0))

    return intercept, coef_log_A, coef_log_D, multiplier


def get_leaf_area_params():
    """Debug helper"""
    return _load_params()


def leaf_area_from_proj(A_proj_cm2: float, D_proj_cm: float) -> float:
    """
    Predict destructive leaf area (cm^2) from projected area + diameter using:
      log(y) = b0 + bA*log(A) + bD*log(D)
      y = exp(log(y))
    """
    if A_proj_cm2 is None or D_proj_cm is None:
        return 0.0
    if A_proj_cm2 <= 0 or D_proj_cm <= 0:
        return 0.0

    b0, bA, bD, mult = _load_params()
    y_log = b0 + bA * math.log(A_proj_cm2) + bD * math.log(D_proj_cm)
    return float(math.exp(y_log) * mult)
