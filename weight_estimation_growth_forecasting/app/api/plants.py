from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

from app.core.db_deps import get_db
from app.core.db_models import PredictionLog, PlantScan, PlantMeta, GrowthPredictionLog
from app.schemas import PlantListItem, PlantDeleteResponse

router = APIRouter(prefix="/plants", tags=["plants"])

HARVEST_READY_WEIGHT_G = 300.0


@router.get("", response_model=list[PlantListItem])
def list_plants(
    filter: str = Query("all", pattern="^(all|growing|harvest_ready)$"),
    zone_id: str | None = None,
    db: Session = Depends(get_db),
):
    # latest scan/log per plant (PredictionLog) - exclude deleted
    q_logs = db.query(PredictionLog).filter(PredictionLog.deleted_at.is_(None))
    if zone_id:
        q_logs = q_logs.filter(PredictionLog.zone_id == zone_id)

    logs = q_logs.order_by(PredictionLog.plant_id, desc(PredictionLog.ts)).all()

    latest_by_plant: Dict[Tuple[str, str], PredictionLog] = {}
    for r in logs:
        key = (r.plant_id, r.zone_id or "")
        if key not in latest_by_plant:
            latest_by_plant[key] = r

    # Plants from PlantMeta (includes growth prediction-only plants)
    q_meta = db.query(PlantMeta)
    if zone_id:
        q_meta = q_meta.filter(PlantMeta.zone_id == zone_id)
    metas = q_meta.all()
    meta_keys = {(m.plant_id, m.zone_id or "") for m in metas}

    # Combine both sources (scans + meta)
    all_keys = set(latest_by_plant.keys()) | meta_keys

    out: list[PlantListItem] = []

    for plant_id, z in sorted(all_keys):
        latest = latest_by_plant.get((plant_id, z))

        # meta (for age)
        meta = (
            db.query(PlantMeta)
            .filter(
                PlantMeta.plant_id == plant_id,
                PlantMeta.zone_id == (z if z else (zone_id or "z01")),
            )
            .first()
        )

        # ✅ age from meta if exists
        if meta:
            age_days = max(0, (datetime.utcnow().date() - meta.planted_at.date()).days)
        else:
            first_log = (
                db.query(PredictionLog)
                .filter(
                    PredictionLog.plant_id == plant_id,
                    PredictionLog.deleted_at.is_(None)
                )
                .order_by(PredictionLog.ts.asc())
                .first()
            )
            age_days = max(0, (datetime.utcnow().date() - first_log.ts.date()).days) if first_log else 0

        # ✅ if no PredictionLog, use latest GrowthPredictionLog to show predicted values
        pred = None
        if latest is None:
            pred = (
                db.query(GrowthPredictionLog)
                .filter(GrowthPredictionLog.plant_id == plant_id)
                .order_by(desc(GrowthPredictionLog.id))
                .first()
            )

        if latest is not None:
            weight = float(latest.weight_est_g or 0.0)
            area = float(getattr(latest, "A_leaf_est_cm2", 0.0) or 0.0)
            diameter = float(getattr(latest, "D_proj_cm", 0.0) or 0.0)
        else:
            weight = float(getattr(pred, "predicted_weight_g", 0.0) or 0.0) if pred else 0.0
            area = float(getattr(pred, "predicted_area_cm2", 0.0) or 0.0) if pred else 0.0
            diameter = float(getattr(pred, "predicted_diameter_cm", 0.0) or 0.0) if pred else 0.0

        status = "HARVEST_READY" if weight >= HARVEST_READY_WEIGHT_G else "NOT_READY"

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
                area_cm2=area,
                diameter_cm=diameter,
                estimated_weight_g=weight,
                status=status,
                image_url=getattr(scan, "rgb_path", None) if scan else None,
            )
        )

    return out


@router.delete("/{plant_id}", response_model=PlantDeleteResponse)
def delete_plant(plant_id: str, db: Session = Depends(get_db)):
    """Soft delete a plant - marks all related records as deleted"""
    plant_id = plant_id.strip()
    now = datetime.now(timezone.utc)

    # Check if plant exists
    meta = db.query(PlantMeta).filter(PlantMeta.plant_id == plant_id).first()
    logs = db.query(PredictionLog).filter(
        PredictionLog.plant_id == plant_id,
        PredictionLog.deleted_at.is_(None)
    ).all()
    scans = db.query(PlantScan).filter(
        PlantScan.plant_id == plant_id,
        PlantScan.deleted_at.is_(None)
    ).all()

    if not meta and not logs and not scans:
        raise HTTPException(status_code=404, detail=f"Plant {plant_id} not found")

    # ✅ Soft delete: Set deleted_at timestamp
    # Mark all PredictionLog entries as deleted
    db.query(PredictionLog).filter(
        PredictionLog.plant_id == plant_id
    ).update({"deleted_at": now, "updated_at": now})
    
    # Mark all PlantScan entries as deleted
    db.query(PlantScan).filter(
        PlantScan.plant_id == plant_id
    ).update({"deleted_at": now})
    
    # Delete PlantMeta (or could add deleted_at field if needed)
    if meta:
        db.delete(meta)

    db.commit()

    return PlantDeleteResponse(ok=True, plant_id=plant_id, deleted_at=now.isoformat())