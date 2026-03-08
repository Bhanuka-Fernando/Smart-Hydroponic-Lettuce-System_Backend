import json
from typing import List, Optional

from app.core.db import SessionLocal
from app.core.db_models import ScanLog
from app.schemas import LogCreate, LogItem, RecentActivityItem

def insert_log(payload: LogCreate) -> int:
    db = SessionLocal()
    try:
        row = ScanLog(
            plant_id=payload.plant_id,
            captured_at=payload.captured_at,
            health_score=payload.health_score,
            status=payload.status,
            main_issue=payload.main_issue,
            probs_json=json.dumps(payload.probs),
            tipburn_json=json.dumps(payload.tipburn),
            image_name=payload.image_name,
            image_path=payload.image_path,
            reason=payload.reason,
            classification_label=payload.classification_label,
            classification_confidence=payload.classification_confidence,
            raw_result_json=json.dumps(payload.raw_result) if payload.raw_result else None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()

def get_logs_for_plant(plant_id: str, limit: int = 50) -> List[LogItem]:
    db = SessionLocal()
    try:
        rows = (
            db.query(ScanLog)
            .filter(ScanLog.plant_id == plant_id)
            .order_by(ScanLog.captured_at.desc())
            .limit(limit)
            .all()
        )
        return [
            LogItem(
                id=r.id,
                plant_id=r.plant_id,
                captured_at=r.captured_at,
                health_score=r.health_score,
                status=r.status,
                main_issue=r.main_issue,
                image_name=r.image_name,
                image_path=r.image_path,
                reason=r.reason,
                classification_label=r.classification_label,
                classification_confidence=r.classification_confidence,
            )
            for r in rows
        ]
    finally:
        db.close()

def get_latest_for_plant(plant_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        r = (
            db.query(ScanLog)
            .filter(ScanLog.plant_id == plant_id)
            .order_by(ScanLog.captured_at.desc())
            .first()
        )
        if not r:
            return None

        return {
            "id": r.id,
            "plant_id": r.plant_id,
            "captured_at": r.captured_at,
            "health_score": r.health_score,
            "status": r.status,
            "main_issue": r.main_issue,
            "image_name": r.image_name,
            "image_path": r.image_path,
            "reason": r.reason,
            "classification_label": r.classification_label,
            "classification_confidence": r.classification_confidence,
            "probs": json.loads(r.probs_json) if r.probs_json else {},
            "tipburn": json.loads(r.tipburn_json) if r.tipburn_json else {},
            "raw_result": json.loads(r.raw_result_json) if r.raw_result_json else None,
        }
    finally:
        db.close()

def get_critical_recent(limit: int = 5) -> List[RecentActivityItem]:
    db = SessionLocal()
    try:
        rows = (
            db.query(ScanLog)
            .filter(ScanLog.status == "ACT NOW")
            .order_by(ScanLog.captured_at.desc())
            .limit(limit)
            .all()
        )
        return [
            RecentActivityItem(
                id=r.id,
                plant_id=r.plant_id,
                captured_at=r.captured_at,
                health_score=r.health_score,
                status=r.status,
                main_issue=r.main_issue,
                image_name=r.image_name,
                image_path=r.image_path,
                reason=r.reason,
                classification_label=r.classification_label,
                classification_confidence=r.classification_confidence,
            )
            for r in rows
        ]
    finally:
        db.close()

def get_all_logs(limit: int = 50, offset: int = 0):
    db = SessionLocal()
    try:
        rows = (
            db.query(ScanLog)
            .order_by(ScanLog.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return [
            LogItem(
                id=r.id,
                plant_id=r.plant_id,
                captured_at=r.captured_at,
                health_score=r.health_score,
                status=r.status,
                main_issue=r.main_issue,
                image_name=r.image_name,
            )
            for r in rows
        ]
    finally:
        db.close()

def get_log_by_id(log_id: int) -> Optional[dict]:
    db = SessionLocal()
    try:
        r = db.query(ScanLog).filter(ScanLog.id == log_id).first()
        if not r:
            return None

        return {
            "id": r.id,
            "plant_id": r.plant_id,
            "captured_at": r.captured_at,
            "health_score": r.health_score,
            "status": r.status,
            "main_issue": r.main_issue,
            "image_name": r.image_name,
            "image_path": r.image_path,
            "reason": r.reason,
            "classification_label": r.classification_label,
            "classification_confidence": r.classification_confidence,
            "probs": json.loads(r.probs_json) if r.probs_json else {},
            "tipburn": json.loads(r.tipburn_json) if r.tipburn_json else {},
            "raw_result": json.loads(r.raw_result_json) if r.raw_result_json else None,
        }
    finally:
        db.close()

    def get_all_logs(limit: int = 50, offset: int = 0):
        db = SessionLocal()
        try:
            rows = (
                db.query(ScanLog)
                .order_by(ScanLog.id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return [
                LogItem(
                    id=r.id,
                    plant_id=r.plant_id,
                    captured_at=r.captured_at,
                    health_score=r.health_score,
                    status=r.status,
                    main_issue=r.main_issue,
                    image_name=r.image_name,
                )
                for r in rows
            ]
        finally:
            db.close()