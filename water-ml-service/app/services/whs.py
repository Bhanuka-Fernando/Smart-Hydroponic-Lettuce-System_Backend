# app/services/whs.py

def _score_in_range(x: float, lo: float, hi: float, warn_lo: float, warn_hi: float) -> float:
    if lo <= x <= hi:
        return 100.0
    if warn_lo <= x < lo:
        return 60.0 + 40.0 * (x - warn_lo) / (lo - warn_lo)
    if hi < x <= warn_hi:
        return 60.0 + 40.0 * (warn_hi - x) / (warn_hi - hi)
    return 20.0


def compute_whs(pH: float, EC_mS_cm: float, temp_C: float, do_mg_L: float):
    # thresholds (tweak later if needed)
    ph_s = _score_in_range(pH,        lo=5.5, hi=6.5, warn_lo=5.2, warn_hi=6.8)
    ec_s = _score_in_range(EC_mS_cm,  lo=1.0, hi=2.0, warn_lo=0.7, warn_hi=2.4)
    t_s  = _score_in_range(temp_C,    lo=18,  hi=26,  warn_lo=16,  warn_hi=28)
    do_s = _score_in_range(do_mg_L,   lo=6.0, hi=9.5, warn_lo=5.0, warn_hi=10.5)

    whs = 0.30 * do_s + 0.25 * ph_s + 0.25 * ec_s + 0.20 * t_s
    whs = round(float(whs), 1)

    if whs >= 80:
        risk = "SAFE"
    elif whs >= 60:
        risk = "WARNING"
    else:
        risk = "CRITICAL"

    return whs, risk
