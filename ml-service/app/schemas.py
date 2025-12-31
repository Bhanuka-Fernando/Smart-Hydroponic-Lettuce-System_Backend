from pydantic import BaseModel
from typing import Optional

class Sensors(BaseModel):
    airT_mean_3d_C: float
    RH_mean_3d_pct: float
    EC_mean_3d_mScm: float
    pH_mean_3d: float

class InferRequest(BaseModel):
    dap: int
    sensors: Sensors
    # optional if your growth model needs them
    deltaA_cm2: Optional[float] = 0.0
    RGR: Optional[float] = 0.0

class InferResponse(BaseModel):
    A_proj_cm2: float
    D_proj_cm: float
    A_des_cm2: float
    W_today_g: float
    A_proj_tmr_cm2: float
    D_proj_tmr_cm: float
    W_tmr_g: float
