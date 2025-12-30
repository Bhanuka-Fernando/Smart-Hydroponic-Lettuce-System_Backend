# ml-service/app/services/growth.py

import pickle
import numpy as np

try:
    import joblib
except Exception:
    joblib = None

from app.core.paths import GROWTH_BUNDLE


def _load_bundle(path: str):
    # Prefer joblib for sklearn artifacts
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    # Fallback: pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _pick_predictor(bundle):
    """
    Return the first object that has a .predict method.
    Handles: bundle being dict, pipeline, estimator, etc.
    """
    candidates = []

    if isinstance(bundle, dict):
        # common keys first
        for k in ("model", "estimator", "regressor", "pipeline", "sk_model"):
            if k in bundle:
                candidates.append(bundle[k])
        # also scan all values
        candidates.extend(list(bundle.values()))
    else:
        candidates.append(bundle)

    for obj in candidates:
        if hasattr(obj, "predict"):
            return obj

    # If we got here: nothing had predict()
    info = []
    if isinstance(bundle, dict):
        for k, v in bundle.items():
            info.append(f"{k}: {type(v)}")
        info_str = ", ".join(info)
        raise RuntimeError(
            "Loaded growth bundle, but couldn't find any object with .predict(). "
            f"Bundle keys/types: {info_str}"
        )
    raise RuntimeError(
        f"Loaded growth bundle of type {type(bundle)}, but it has no .predict(). "
        "This usually means you saved the wrong thing (like coefficients / arrays)."
    )


BUNDLE = _load_bundle(GROWTH_BUNDLE)
MODEL = _pick_predictor(BUNDLE)

# feature order (optional but recommended)
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
        # fallback (not ideal)
        X = np.array([[row[k] for k in row.keys()]], dtype=float)

    y = MODEL.predict(X)[0]   # expecting [A_tmr, D_tmr]
    return float(y[0]), float(y[1])
