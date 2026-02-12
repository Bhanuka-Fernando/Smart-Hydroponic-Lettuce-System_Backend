from sqlalchemy import Column, Integer, String, Float, DateTime
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
