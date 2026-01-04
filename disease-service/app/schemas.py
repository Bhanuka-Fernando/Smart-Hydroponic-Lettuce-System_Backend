from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class TipburnOut(BaseModel):
    num_boxes: int
    A: float
    C: float

class PredictResponse(BaseModel):
    plant_id: str
    captured_at: str
    probs: Dict[str, float]
    tipburn: TipburnOut
    health_score: int
    status: str
    main_issue: str
    risk_cls: float
    risk_tip: float
    risk_total: float
    tipburn_present: bool
    tipburn_A: float
    tipburn_C: float
    tipburn_A_cap: float

class LogCreate(BaseModel):
    plant_id: str
    captured_at: str
    health_score: int
    status: str
    main_issue: str
    probs: Dict[str, float]
    tipburn: Dict[str, Any]
    image_name: Optional[str] = None

class LogItem(BaseModel):
    id: int
    plant_id: str
    captured_at: str
    health_score: int
    status: str
    main_issue: str
    image_name: Optional[str] = None

class RecentActivityItem(LogItem):
    pass
