from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class WaterReading(BaseModel):
    ph: float
    temp_c: float
    turb_ntu: float
    ec_or_tds: Optional[float] = None
    mode: str = "EC"  # "EC" or "TDS"

class WaterReadingTS(WaterReading):
    timestamp: datetime  # ISO string accepted

class WaterBatchRequest(BaseModel):
    pond_id: Optional[str] = "POND_01"
    readings: List[WaterReadingTS] = Field(min_length=2)

class WaterStatusResponse(BaseModel):
    status_model: str
    proba_ok: float
    proba_warn: float
    proba_crit: float
    health_score: float
    score_status: str
    final_status: str
    reasons: List[str]
    actions: List[str]
    meta: Dict[str, str] = Field(default_factory=dict)
