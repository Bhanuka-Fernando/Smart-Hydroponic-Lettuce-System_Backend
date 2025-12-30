import json, math
from app.core.paths import LEAF_AREA_JSON

PARAMS = json.loads(LEAF_AREA_JSON.read_text())

INTERCEPT = float(PARAMS["intercept"])
B_A = float(PARAMS["coef_log_A"])
B_D = float(PARAMS["coef_log_D"])

def leaf_area_from_proj(A_proj_cm2: float, D_proj_cm: float) -> float:
    A_proj_cm2 = max(A_proj_cm2, 1e-6)
    D_proj_cm = max(D_proj_cm, 1e-6)
    lnA = INTERCEPT + B_A * math.log(A_proj_cm2) + B_D * math.log(D_proj_cm)
    return float(math.exp(lnA))
