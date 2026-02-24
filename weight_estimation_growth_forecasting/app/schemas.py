from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime

class Sensors(BaseModel):
    airT: Optional[float] = None
    RH: Optional[float] = None
    EC: Optional[float] = None
    pH: Optional[float] = None

class InferRequest(BaseModel):
    plant_id: str
    zone_id: str
    dap: int
    sensors: Optional[Sensors] = None
    A_prev_cm2: Optional[float] = None

class InferResponse(BaseModel):
    A_proj_cm2: float
    D_proj_cm: float
    A_des_cm2: float
    W_today_g: float
    A_proj_tmr_cm2: float
    D_proj_tmr_cm: float
    W_tmr_g: float
    mask_overlay_b64: Optional[str] = None

class ForecastRequest(BaseModel):
    plant_id: str
    zone_id: str
    dap: int
    n_days: int
    A_prev_cm2: Optional[float] = None 
    A_t_cm2: float
    D_t_cm: float
    sensors: Optional[Any] = None

class ForecastPoint(BaseModel):
    step: int
    DAP_pred: int
    A_pred_cm2: float
    D_pred_cm: float
    A_leaf_pred_cm2: float
    W_pred_g: float

class ForecastResponse(BaseModel):
    points: List[ForecastPoint]


class PanelTodayResponse(BaseModel):
    Leaf_Area_today_cm2: float
    Diameter_today_cm: float
    Weight_today_g: float

    Leaf_Area_tomorrow_cm2: float
    Diameter_tomorrow_cm: float
    Weight_tomorrow_g: float


## IOT SIMULATION

class SensorPacket(BaseModel):
    device_id: str
    zone_id: str
    ts: datetime
    airT: float
    RH: float
    EC: float
    pH: float

class LatestDashboardResponse(BaseModel):
    zone_id: str
    plant_id: str
    ts: datetime | None = None

    # latest sensors (raw)
    airT: float | None = None
    RH: float | None = None
    EC: float | None = None
    pH: float | None = None

    # 3-day means
    airT_mean_3d_C: float | None = None
    RH_mean_3d_pct: float | None = None
    EC_mean_3d_mScm: float | None = None
    pH_mean_3d: float | None = None

    # predictions
    A_proj_cm2: float | None = None
    D_proj_cm: float | None = None
    A_leaf_est_cm2: float | None = None
    weight_est_g: float | None = None
    A_next_cm2: float | None = None
    D_next_cm: float | None = None