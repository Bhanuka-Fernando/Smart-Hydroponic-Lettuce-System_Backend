from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.core.db_deps import get_db
from app.core.db_models import PredictionLog, GrowthPrediction
from app.schemas import GrowthPredictionSaveRequest, GrowthPredictionSaveResponse

router = APIRouter(prefix="/growth", tags=["growth"])


@router.post("/predict/save", response_model=GrowthPredictionSaveResponse)
def save_growth_prediction(
    prediction: GrowthPredictionSaveRequest,
    db: Session = Depends(get_db),
):
    """
    Save a growth prediction with historical series data.
    User ID should come from authenticated user in production.
    
    Note: Plant existence is not strictly validated - predictions can be made
    for any plant_id even if no weight measurements exist yet.
    """
    # Create the growth prediction record
    growth_pred = GrowthPrediction(
        plant_id=prediction.plant_id,
        user_id=None,  # TODO: Get from authenticated user context
        date_label=prediction.date_label,
        predicted_weight_g=prediction.predicted_weight_g,
        predicted_area_cm2=prediction.predicted_area_cm2,
        predicted_diameter_cm=prediction.predicted_diameter_cm,
        change_pct=prediction.change_pct,
        series_data=prediction.series.model_dump(),
        created_at=datetime.now(timezone.utc)
    )
    
    db.add(growth_pred)
    db.commit()
    db.refresh(growth_pred)
    
    return GrowthPredictionSaveResponse(
        ok=True,
        prediction_id=f"pred_{growth_pred.id}",
        saved_at=growth_pred.created_at.isoformat()
    )
