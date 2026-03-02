# app/models.py
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class SpoilagePrediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    plant_id: str
    captured_at: datetime

    temperature: float
    humidity: float

    stage: str
    status: str
    remaining_days: float

    p_fresh: float
    p_slightly_aged: float
    p_near_spoilage: float
    p_spoiled: float

    image_url: str | None = None