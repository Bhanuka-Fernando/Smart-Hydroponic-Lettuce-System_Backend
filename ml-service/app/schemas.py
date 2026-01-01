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
    A_prev_cm2: Optional[float] = None  # yesterday projected area (from DB)

class InferResponse(BaseModel):
    A_proj_cm2: float
    D_proj_cm: float
    A_des_cm2: float
    W_today_g: float
    A_proj_tmr_cm2: float
    D_proj_tmr_cm: float
    W_tmr_g: float

class ForecastRequest(BaseModel):
    dap: int
    n_days: int
    A_prev_cm2: float
    A_t_cm2: float
    D_t_cm: float
    sensors: Sensors

class ForecastPoint(BaseModel):
    step: int
    DAP_pred: int
    A_pred_cm2: float
    D_pred_cm: float

class ForecastResponse(BaseModel):
    points: list[ForecastPoint]
