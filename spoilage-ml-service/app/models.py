from sqlmodel import SQLModel, Field
from datetime import datetime, timezone
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

    # original dataset image used for simulated scan, e.g. IMG_3681.jpg
    sim_source_image: str | None = None


class SpoilageAlert(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    plant_id: str = Field(index=True)
    prediction_id: Optional[int] = Field(default=None, index=True)

    stage: str
    severity: str  # warning | critical
    title: str
    message: str

    is_acknowledged: bool = Field(default=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
    )
    acknowledged_at: Optional[datetime] = None