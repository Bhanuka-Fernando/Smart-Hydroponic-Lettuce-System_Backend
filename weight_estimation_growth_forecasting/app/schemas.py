from pydantic import BaseModel
from typing import Optional, List, Any, Dict
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



class PlantListItem(BaseModel):
    plant_id: str
    name: str
    age_days: int
    area_cm2: Optional[float] = None
    diameter_cm: Optional[float] = None
    estimated_weight_g: Optional[float] = None
    status: str  # "NOT_READY" | "HARVEST_READY"
    image_url: Optional[str] = None


class PlantHistoryItem(BaseModel):
    date: str
    date_label: str
    actual_weight_g: Optional[float] = None
    predicted_weight_g: Optional[float] = None
    delta_g: Optional[float] = None
    status: str  # "On Track" etc.


class PlantDetailsResponse(BaseModel):
    plant_id: str
    display_name: str
    planted_on: str
    age_days: int
    start_weight_g: float
    current_weight_g: float
    growth_pct: float
    predicted_today_g: Optional[float] = None
    trajectory: Optional[dict] = None  # {labels:[], values:[]}
    history: Optional[List[PlantHistoryItem]] = None

class WeightSaveRequest(BaseModel):
    plant_id: str
    zone_id: str
    captured_at: datetime

    A_proj_cm2: float
    D_proj_cm: float
    A_des_cm2: float
    W_today_g: float

    image_url: Optional[str] = None

class HistoryItem(BaseModel):
    ts: datetime
    A_proj_cm2: float
    D_proj_cm: float
    A_leaf_est_cm2: float
    weight_est_g: float


# -------------------------
# NEW: Frontend-Compatible Schemas
# -------------------------

class DashboardMetricsResponse(BaseModel):
    zone_id: str
    zone_name: str
    plant_count: int
    harvest_ready_count: int
    avg_growth_pct: float
    temperature_c: float
    humidity_pct: float
    ec_ms_cm: float
    ph: float
    last_updated: str


class IoTSensorPayload(BaseModel):
    zone_id: str
    temperature_c: float
    humidity_pct: float
    ec_ms_cm: float
    ph: float
    timestamp: Optional[str] = None


class IoTIngestResponse(BaseModel):
    ok: bool
    sensor_id: str
    recorded_at: str


class ActivityItem(BaseModel):
    id: str
    type: str
    title: str
    description: str
    timestamp: str
    zone: Optional[str] = None
    status: Optional[str] = None


class ActivityHistoryResponse(BaseModel):
    activities: List[ActivityItem]
    total_count: int
    has_more: bool


# -------------------------
# Growth Prediction Schemas
# -------------------------

class GrowthPredictionSeries(BaseModel):
    labels: List[str]
    actual: List[float]
    predicted: List[float]


class GrowthPredictionSaveRequest(BaseModel):
    plant_id: str
    date_label: str
    predicted_weight_g: float
    predicted_area_cm2: float
    predicted_diameter_cm: float
    change_pct: float
    series: GrowthPredictionSeries


class GrowthPredictionSaveResponse(BaseModel):
    ok: bool
    prediction_id: str
    saved_at: str


class PlantDeleteResponse(BaseModel):
    ok: bool
    plant_id: str
    deleted_at: str




class GrowthSeries(BaseModel):
    labels: List[str]
    actual: List[float]
    predicted: List[float]

class GrowthPredictSaveRequest(BaseModel):
    plant_id: str
    date_label: str
    predicted_weight_g: float
    predicted_area_cm2: float
    predicted_diameter_cm: float
    change_pct: float = 0.0
    series: Optional[GrowthSeries] = None
    insight: Optional[Dict[str, Any]] = None
