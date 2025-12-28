from pydantic import BaseModel
from typing import Dict

class SpoilageClassifyResponse(BaseModel):
    stage: str
    confidence: float
    probs: Dict[str, float]
