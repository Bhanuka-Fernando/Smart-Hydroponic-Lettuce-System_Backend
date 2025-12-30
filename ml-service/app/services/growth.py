import pickle
import numpy as np
from app.core.paths import GROWTH_BUNDLE

with open(GROWTH_BUNDLE, "rb") as f:
    BUNDLE = pickle.load(f)

MODEL = BUNDLE["model"] if isinstance(BUNDLE, dict) and "model" in BUNDLE else BUNDLE
FEATURES = None
if isinstance(BUNDLE, dict):
    for k in ("feature_cols", "feature_columns", "feature_names", "columns", "feature_col_order"):
        if k in BUNDLE:
            FEATURES = list(BUNDLE[k])
            break


def predict_tomorrow(dap, A_t_cm2, D_t_cm, deltaA_cm2, RGR, sensors):
    row = {
        "DAP": dap,
        "A_t_cm2": A_t_cm2,
        "D_t_cm": D_t_cm,
        "deltaA_cm2": deltaA_cm2,
        "RGR": RGR,
        "airT_mean_3d_C": sensors.airT_mean_3d_C,
        "RH_mean_3d_pct": sensors.RH_mean_3d_pct,
        "EC_mean_3d_mScm": sensors.EC_mean_3d_mScm,
        "pH_mean_3d": sensors.pH_mean_3d,
    }

    if FEATURES:
        X = np.array([[row[c] for c in FEATURES]], dtype=float)
    else:
        # Only if you don't have saved order (not ideal)
        X = np.array([[row[k] for k in row.keys()]], dtype=float)

    y = MODEL.predict(X)[0]
    return float(y[0]), float(y[1])
