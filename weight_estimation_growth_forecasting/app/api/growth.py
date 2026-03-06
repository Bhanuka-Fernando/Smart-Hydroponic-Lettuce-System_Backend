from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from app.core.db_deps import get_db
from app.core.db_models import GrowthPredictionLog, PlantMeta
from app.schemas import GrowthPredictSaveRequest

router = APIRouter(prefix="/infer/growth", tags=["growth"])


@router.post("/predict/save")
def save_growth_prediction(
    payload: GrowthPredictSaveRequest,
    db: Session = Depends(get_db),
):
    """
    Save a growth prediction with series data and update plant metadata.
    This creates a GrowthPredictionLog entry and updates/creates PlantMeta
    with calculated planted_at date based on age_days.
    """
    # 1) Save growth prediction log
    row = GrowthPredictionLog(
        plant_id=payload.plant_id,
        date_label=payload.date_label,
        predicted_weight_g=float(payload.predicted_weight_g),
        predicted_area_cm2=float(payload.predicted_area_cm2),
        predicted_diameter_cm=float(payload.predicted_diameter_cm),
        change_pct=float(payload.change_pct or 0.0),
        series=payload.series.model_dump() if payload.series else None,
        insight=payload.insight,
    )
    db.add(row)

    # 2) Update PlantMeta.planted_at based on age_days
    # planted_at = today - age_days
    planted_at = datetime.utcnow() - timedelta(days=int(payload.age_days))

    meta = (
        db.query(PlantMeta)
        .filter(PlantMeta.plant_id == payload.plant_id, PlantMeta.zone_id == payload.zone_id)
        .first()
    )

    if meta is None:
        meta = PlantMeta(
            plant_id=payload.plant_id,
            zone_id=payload.zone_id,
            planted_at=planted_at,
            updated_at=datetime.utcnow(),
        )
        db.add(meta)
    else:
        meta.planted_at = planted_at
        meta.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(row)
    
    return {
        "ok": True,
        "prediction_id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/predictions/{plant_id}")
def get_plant_predictions(
    plant_id: str,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Get all growth predictions for a specific plant, ordered by most recent first.
    Returns prediction history with series data.
    """
    predictions = (
        db.query(GrowthPredictionLog)
        .filter(GrowthPredictionLog.plant_id == plant_id)
        .order_by(desc(GrowthPredictionLog.created_at))
        .limit(limit)
        .all()
    )

    if not predictions:
        return {
            "plant_id": plant_id,
            "predictions": [],
            "count": 0,
        }

    return {
        "plant_id": plant_id,
        "predictions": [
            {
                "id": p.id,
                "date_label": p.date_label,
                "predicted_weight_g": p.predicted_weight_g,
                "predicted_area_cm2": p.predicted_area_cm2,
                "predicted_diameter_cm": p.predicted_diameter_cm,
                "change_pct": p.change_pct,
                "series": p.series,
                "insight": p.insight,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in predictions
        ],
        "count": len(predictions),
    }


@router.get("/predictions/{plant_id}/latest")
def get_latest_prediction(
    plant_id: str,
    db: Session = Depends(get_db),
):
    """
    Get the most recent growth prediction for a plant.
    Useful for displaying current forecast in UI.
    """
    prediction = (
        db.query(GrowthPredictionLog)
        .filter(GrowthPredictionLog.plant_id == plant_id)
        .order_by(desc(GrowthPredictionLog.created_at))
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No growth predictions found for plant_id: {plant_id}"
        )

    return {
        "plant_id": plant_id,
        "id": prediction.id,
        "date_label": prediction.date_label,
        "predicted_weight_g": prediction.predicted_weight_g,
        "predicted_area_cm2": prediction.predicted_area_cm2,
        "predicted_diameter_cm": prediction.predicted_diameter_cm,
        "change_pct": prediction.change_pct,
        "series": prediction.series,
        "insight": prediction.insight,
        "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
    }


@router.delete("/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a specific growth prediction by ID.
    This is a hard delete - use with caution.
    """
    prediction = (
        db.query(GrowthPredictionLog)
        .filter(GrowthPredictionLog.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found: {prediction_id}"
        )

    plant_id = prediction.plant_id
    db.delete(prediction)
    db.commit()

    return {
        "ok": True,
        "prediction_id": prediction_id,
        "plant_id": plant_id,
        "deleted_at": datetime.utcnow().isoformat(),
    }
