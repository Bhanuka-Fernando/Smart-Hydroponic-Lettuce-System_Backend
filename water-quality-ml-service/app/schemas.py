from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

# ---------- Requests ----------

class AnalyzeRequest(BaseModel):
    tank_id: str = Field(..., examples=["TANK_01"])
    timestamp: str = Field(..., examples=["2025-11-15T10:30:00Z"])
    ph: float
    temp_c: float
    turb_ntu: float
    ec: float


class Reading(BaseModel):
    timestamp: str
    ph: float
    temp_c: float
    turb_ntu: float
    ec: float


class AnalyzeBatchRequest(BaseModel):
    tank_id: str
    readings: List[Reading]


class IngestRequest(BaseModel):
    tank_id: str
    readings: List[Reading]


# ---------- Responses ----------

class AnalyzeResponse(BaseModel):
    tank_id: str
    timestamp: str
    mode: Literal["single", "batch_timeseries"]

    ml_status: str
    ml_probs: Dict[str, float]

    ml_algae: str
    ml_algae_probs: Dict[str, float]

    rule_status: str
    health_score: int
    score_status: str
    final_status: str

    # ✅ Main one-liner outputs for UI
    main_reason: str
    main_action: str

    # Details (expand/collapsible in UI)
    reasons: List[str]
    actions: List[str]

    algae_reasons: List[str]
    algae_actions: List[str]

    sensor_quality: Literal["OK", "SUSPECT"]
    sensor_notes: List[str] = []

    meta: Dict[str, object] = {}


class IngestResponse(BaseModel):
    saved: int
    tank_id: str


class LatestResponse(BaseModel):
    tank_id: str
    timestamp: str
    ph: float
    temp_c: float
    turb_ntu: float
    ec: float


class HistoryResponse(BaseModel):
    tank_id: str
    count: int
    readings: List[LatestResponse]
