from pydantic import BaseModel, Field
from typing import Dict

class SpoilagePredictResponse(BaseModel):
    stage: str
    stage_probs: Dict[str, float]
    remaining_days: float = Field(ge=0)
    status: str

class StageProbs(BaseModel):
    fresh: float = Field(ge=0, le=1)
    slightly_aged: float = Field(ge=0, le=1)
    near_spoilage: float = Field(ge=0, le=1)
    spoiled: float = Field(ge=0, le=1)
    
class StageOnlyResponse(BaseModel):
    stage: str
    stage_probs: StageProbs
    status: str

class RemainingDaysOnlyRequest(BaseModel):
    stage_probs: Dict[str, float]
    temperature: float
    humidity: float

class RemainingDaysOnlyResponse(BaseModel):
    remaining_days: float = Field(ge=0)
