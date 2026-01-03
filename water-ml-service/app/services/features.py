# app/services/features.py
import pandas as pd

def build_features(df: pd.DataFrame, w: int = 3) -> pd.DataFrame:
    """
    Builds features that match feature_list.json:
      <col>_mean, <col>_std, d<col> (delta vs previous row)
    Needs columns: timestamp, tank_id, pH, EC_mS_cm, temp_C, do_mg_L
    """
    df = df.sort_values(["tank_id", "timestamp"]).copy()

    def _feat(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        cols = ["pH", "EC_mS_cm", "temp_C", "do_mg_L"]

        # rolling mean/std
        for c in cols:
            g[f"{c}_mean"] = g[c].rolling(w, min_periods=w).mean()
            g[f"{c}_std"]  = g[c].rolling(w, min_periods=w).std()

        # deltas (previous step)
        g["dpH"] = g["pH"] - g["pH"].shift(1)
        g["dEC_mS_cm"] = g["EC_mS_cm"] - g["EC_mS_cm"].shift(1)
        g["dtemp_C"] = g["temp_C"] - g["temp_C"].shift(1)
        g["ddo_mg_L"] = g["do_mg_L"] - g["do_mg_L"].shift(1)

        # extra feature (you already had it)
        g["temp_do_risk"] = g["temp_C"] / (g["do_mg_L"].replace(0, 0.001))

        return g

    return df.groupby("tank_id", group_keys=False).apply(_feat)
