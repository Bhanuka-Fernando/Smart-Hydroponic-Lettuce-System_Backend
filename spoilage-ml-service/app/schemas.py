from pydantic import BaseModel, Field
from typing import Optional

PLANT_ID_PATTERN = r"^(SIM-)?P-\d{3}$"


class StageProbs(BaseModel):
    fresh: float = Field(ge=0, le=1)
    slightly_aged: float = Field(ge=0, le=1)
    near_spoilage: float = Field(ge=0, le=1)
    spoiled: float = Field(ge=0, le=1)


class SpoilagePredictResponse(BaseModel):
    plant_id: str = Field(..., pattern=PLANT_ID_PATTERN)
    captured_at: str
    stage: str
    stage_probs: StageProbs
    remaining_days: float = Field(ge=0)
    status: str


class StageOnlyResponse(BaseModel):
    plant_id: str = Field(..., pattern=PLANT_ID_PATTERN)
    captured_at: str
    stage: str
    stage_probs: StageProbs
    status: str


class RemainingDaysOnlyRequest(BaseModel):
    plant_id: Optional[str] = Field(default=None, pattern=PLANT_ID_PATTERN)
    captured_at: Optional[str] = None
    stage_probs: StageProbs
    temperature: float
    humidity: float


class RemainingDaysOnlyResponse(BaseModel):
    plant_id: Optional[str] = None
    captured_at: Optional[str] = None
    remaining_days: float = Field(ge=0)