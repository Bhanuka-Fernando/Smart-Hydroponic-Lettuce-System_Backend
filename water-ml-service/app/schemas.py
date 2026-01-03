from pydantic import BaseModel, Field
from typing import List, Literal, Optional

RiskLevel = Literal["SAFE", "WARNING", "CRITICAL"]
Priority = Literal["LOW", "MED", "HIGH"]

class Reading(BaseModel):
    timestamp: str
    pH: float
    EC_mS_cm: float
    temp_C: float
    do_mg_L: float

class AnalyzeRequest(BaseModel):
    tank_id: str = Field(..., examples=["TANK_01"])
    readings: List[Reading] = Field(..., description="Last 30–60 minutes readings (e.g., 6 points for 10-min sampling).")

class Recommendation(BaseModel):
    priority: Priority
    issue: str
    actions: List[str]
    reason: str

class AnalyzeResponse(BaseModel):
    tank_id: str
    timestamp: str
    WHS_0_100: float
    risk_level: RiskLevel
    anom_score_0_1: float
    threshold: float
    anomaly_flag: int
    persistence: str
    recommendations: List[Recommendation]
