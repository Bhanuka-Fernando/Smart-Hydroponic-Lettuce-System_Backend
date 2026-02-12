from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.db_models import SensorReading

def get_3day_means(db: Session, zone_id: str, now: datetime):
    start = now - timedelta(days=3)

    airT, RH, EC, pH = db.query(
        func.avg(SensorReading.airT),
        func.avg(SensorReading.RH),
        func.avg(SensorReading.EC),
        func.avg(SensorReading.pH),
    ).filter(
        SensorReading.zone_id == zone_id,
        SensorReading.ts >= start
    ).one()

    if airT is None:
        return None

    return {
        "airT_mean_3d_C": float(airT),
        "RH_mean_3d_pct": float(RH),
        "EC_mean_3d_mScm": float(EC),
        "pH_mean_3d": float(pH),
    }
