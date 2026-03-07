from sqlalchemy import Column, Integer, String, Text, Index
from app.core.db import Base

class ScanLog(Base):
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(String, index=True, nullable=False)
    captured_at = Column(String, index=True, nullable=False)
    health_score = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    main_issue = Column(String, nullable=False)
    probs_json = Column(Text, nullable=False)
    tipburn_json = Column(Text, nullable=False)
    image_name = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_logs_plant_time", "plant_id", "captured_at"),
        Index("idx_logs_time", "captured_at"),
    )