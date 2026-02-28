from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from typing import Optional

from app.core.db_deps import get_db
from app.core.db_models import SensorReading, PredictionLog
from app.schemas import DashboardMetricsResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/latest", response_model=DashboardMetricsResponse)
def dashboard_latest(zone_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Frontend-compatible dashboard metrics endpoint (without /infer prefix).
    Returns latest sensor readings and plant statistics for the specified zone.
    """
    # Get latest sensor reading
    sensor_query = db.query(SensorReading)
    if zone_id:
        sensor_query = sensor_query.filter(SensorReading.zone_id == zone_id)
    latest_sensor = sensor_query.order_by(desc(SensorReading.ts)).first()
    
    # Count total plants in zone (excluding deleted)
    plant_query = db.query(PredictionLog.plant_id).filter(
        PredictionLog.deleted_at.is_(None)
    ).distinct()
    if zone_id:
        plant_query = plant_query.filter(PredictionLog.zone_id == zone_id)
    plant_count = plant_query.count()
    
    # Get latest weight for each plant to determine harvest ready status
    subquery = db.query(
        PredictionLog.plant_id,
        PredictionLog.weight_est_g
    ).filter(
        PredictionLog.deleted_at.is_(None)
    ).distinct(PredictionLog.plant_id)
    if zone_id:
        subquery = subquery.filter(PredictionLog.zone_id == zone_id)
    subquery = subquery.order_by(PredictionLog.plant_id, desc(PredictionLog.ts))
    
    latest_weights = subquery.all()
    harvest_ready_count = sum(1 for _, weight in latest_weights if weight and weight >= 300)
    
    # Calculate average growth percentage
    avg_growth = 0.0
    if latest_weights:
        growth_values = []
        for plant_id, _ in latest_weights:
            logs = db.query(PredictionLog).filter(
                PredictionLog.plant_id == plant_id,
                PredictionLog.deleted_at.is_(None)
            ).order_by(PredictionLog.ts.asc()).limit(2).all()
            
            if len(logs) >= 2:
                start_w = logs[0].weight_est_g or 0
                current_w = logs[-1].weight_est_g or 0
                if start_w > 0:
                    growth_pct = ((current_w - start_w) / start_w) * 100
                    growth_values.append(growth_pct)
        
        if growth_values:
            avg_growth = sum(growth_values) / len(growth_values)
    
    zone_name = f"Zone {zone_id.upper()}" if zone_id else "All Zones"
    last_updated = latest_sensor.ts.isoformat() if latest_sensor else datetime.utcnow().isoformat()
    
    return DashboardMetricsResponse(
        zone_id=zone_id or "all",
        zone_name=zone_name,
        plant_count=plant_count,
        harvest_ready_count=harvest_ready_count,
        avg_growth_pct=round(avg_growth, 1),
        temperature_c=latest_sensor.airT if latest_sensor else 0.0,
        humidity_pct=latest_sensor.RH if latest_sensor else 0.0,
        ec_ms_cm=latest_sensor.EC if latest_sensor else 0.0,
        ph=latest_sensor.pH if latest_sensor else 0.0,
        last_updated=last_updated,
    )
