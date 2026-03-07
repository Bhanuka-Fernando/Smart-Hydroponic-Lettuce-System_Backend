from __future__ import annotations

from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, DateTime, Integer, Index

class Base(DeclarativeBase):
    pass

class WaterReading(Base):
    __tablename__ = "water_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    tank_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    ph: Mapped[float] = mapped_column(Float)
    temp_c: Mapped[float] = mapped_column(Float)
    turb_ntu: Mapped[float] = mapped_column(Float)
    ec: Mapped[float] = mapped_column(Float)

Index("ix_tank_time", WaterReading.tank_id, WaterReading.timestamp)