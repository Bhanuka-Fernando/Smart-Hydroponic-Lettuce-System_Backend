from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from datetime import datetime, timezone
from app.core.db import Base

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, index=True)

    device_id = Column(String, index=True)
    zone_id = Column(String, index=True)
    ts = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc))

    airT = Column(Float)
    RH = Column(Float)
    EC = Column(Float)
    pH = Column(Float)

class PlantScan(Base):
    __tablename__ = "plant_scans"
    id = Column(Integer, primary_key=True, index=True)

    device_id = Column(String, index=True)
    plant_id = Column(String, index=True)
    zone_id = Column(String, index=True)
    ts = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc))

    rgb_path = Column(String)
    depth_path = Column(String, nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)

    plant_id = Column(String, index=True)
    zone_id = Column(String, index=True)
    ts = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc))

    A_proj_cm2 = Column(Float)
    D_proj_cm = Column(Float)
    A_leaf_est_cm2 = Column(Float)

    weight_est_g = Column(Float)
    A_next_cm2 = Column(Float, nullable=True)
    D_next_cm = Column(Float, nullable=True)
    
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Activity(Base):
    __tablename__ = "activities"
    id = Column(Integer, primary_key=True, index=True)
    
    activity_type = Column(String(50), index=True)
    title = Column(String(200))
    description = Column(Text, nullable=True)
    zone_id = Column(String, index=True, nullable=True)
    user_id = Column(Integer, nullable=True)
    status = Column(String(20), nullable=True)
    meta_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc))


class GrowthPrediction(Base):
    __tablename__ = "growth_predictions"
    id = Column(Integer, primary_key=True, index=True)
    
    plant_id = Column(String(50), index=True)
    user_id = Column(Integer, nullable=True)
    date_label = Column(String(50), nullable=True)
    predicted_weight_g = Column(Float, nullable=True)
    predicted_area_cm2 = Column(Float, nullable=True)
    predicted_diameter_cm = Column(Float, nullable=True)
    change_pct = Column(Float, nullable=True)
    series_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc))

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime

class GrowthPredictionLog(Base):
    __tablename__ = "growth_prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    date_label = Column(String, nullable=False)
    predicted_weight_g = Column(Float, nullable=False)
    predicted_area_cm2 = Column(Float, nullable=False)
    predicted_diameter_cm = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)

    series = Column(JSON, nullable=True)
    insight = Column(JSON, nullable=True)

class PlantMeta(Base):
    __tablename__ = "plant_meta"

    plant_id = Column(String, primary_key=True)
    zone_id = Column(String, primary_key=True)
    planted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    start_weight_g = Column(Float, nullable=True)  # Set ONCE on first scan, never updated
    current_weight_g = Column(Float, nullable=True)  # Updated on every scan

