from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timezone
from typing import Optional, List

from app.core.db_deps import get_db
from app.core.db_models import PredictionLog, PlantScan
from app.schemas import PlantListItem, PlantDeleteResponse

router = APIRouter(prefix="/plants", tags=["plants"])

HARVEST_READY_WEIGHT_G = 300.0


@router.get("", response_model=list[PlantListItem])
def list_plants(
    filter: str = Query("all", pattern="^(all|growing|harvest_ready)$"),
    zone_id: str | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(PredictionLog).filter(PredictionLog.deleted_at.is_(None))
    if zone_id:
        q = q.filter(PredictionLog.zone_id == zone_id)

    latest_logs = q.order_by(PredictionLog.plant_id, desc(PredictionLog.ts)).all()

    latest_by_plant = {}
    for row in latest_logs:
        if row.plant_id not in latest_by_plant:
            latest_by_plant[row.plant_id] = row

    out: list[PlantListItem] = []

    for plant_id, log in latest_by_plant.items():
        first_log = (
            db.query(PredictionLog)
            .filter(PredictionLog.plant_id == plant_id)
            .order_by(PredictionLog.ts.asc())
            .first()
        )
        age_days = max(0, (datetime.utcnow().date() - first_log.ts.date()).days) if first_log else 0

        weight = float(log.weight_est_g) if log.weight_est_g is not None else None
        status = "HARVEST_READY" if (weight is not None and weight >= HARVEST_READY_WEIGHT_G) else "NOT_READY"

        if filter == "growing" and status != "NOT_READY":
            continue
        if filter == "harvest_ready" and status != "HARVEST_READY":
            continue

        scan = (
            db.query(PlantScan)
            .filter(PlantScan.plant_id == plant_id)
            .order_by(desc(PlantScan.ts))
            .first()
        )

        out.append(
            PlantListItem(
                plant_id=plant_id,
                name=f"Plant {plant_id}",
                age_days=age_days,
                area_cm2=float(getattr(log, "A_leaf_est_cm2", 0.0) or 0.0),
                diameter_cm=float(getattr(log, "D_proj_cm", 0.0) or 0.0),
                estimated_weight_g=weight,
                status=status,
                image_url=getattr(scan, "rgb_path", None) if scan else None,
            )
        )

    out.sort(key=lambda x: x.plant_id)
    return out


@router.delete("/{plant_id}", response_model=PlantDeleteResponse)
def delete_plant(
    plant_id: str,
    db: Session = Depends(get_db),
):
    """
    Soft delete a plant by marking all its prediction logs and scans as deleted.
    User authorization should be handled by middleware/dependency in production.
    """
    # Check if plant exists
    existing_logs = db.query(PredictionLog).filter(
        PredictionLog.plant_id == plant_id,
        PredictionLog.deleted_at.is_(None)
    ).first()
    
    if not existing_logs:
        raise HTTPException(status_code=404, detail=f"Plant {plant_id} not found or already deleted")
    
    # Soft delete all prediction logs for this plant
    now = datetime.now(timezone.utc)
    db.query(PredictionLog).filter(
        PredictionLog.plant_id == plant_id
    ).update({
        "deleted_at": now,
        "updated_at": now
    })
    
    # Soft delete all scans for this plant
    db.query(PlantScan).filter(
        PlantScan.plant_id == plant_id
    ).update({
        "deleted_at": now
    })
    
    db.commit()
    
    return PlantDeleteResponse(
        ok=True,
        plant_id=plant_id,
        deleted_at=now.isoformat()
    )