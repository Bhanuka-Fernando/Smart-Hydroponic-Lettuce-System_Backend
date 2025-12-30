from pydantic import BaseModel, Field
from typing import Dict

class SpoilagePredictResponse(BaseModel):
    stage: str
    stage_probs: Dict[str, float]
    remaining_days: float = Field(ge=0)
    status: str
