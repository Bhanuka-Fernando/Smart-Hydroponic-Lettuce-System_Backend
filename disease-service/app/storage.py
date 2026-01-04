import json
from typing import List, Optional

from app.db import get_conn
from app.schemas import LogCreate, LogItem, RecentActivityItem

def insert_log(payload: LogCreate) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO scan_logs
            (plant_id, captured_at, health_score, status, main_issue, probs_json, tipburn_json, image_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.plant_id,
                payload.captured_at,
                payload.health_score,
                payload.status,
                payload.main_issue,
                json.dumps(payload.probs),
                json.dumps(payload.tipburn),
                payload.image_name,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

def get_logs_for_plant(plant_id: str, limit: int = 50) -> List[LogItem]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, plant_id, captured_at, health_score, status, main_issue, image_name
            FROM scan_logs
            WHERE plant_id = ?
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            (plant_id, limit),
        ).fetchall()
    return [LogItem(**dict(r)) for r in rows]

def get_latest_for_plant(plant_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(
            """
            SELECT id, plant_id, captured_at, health_score, status, main_issue, image_name
            FROM scan_logs
            WHERE plant_id = ?
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            (plant_id,),
        ).fetchone()
    return dict(r) if r else None

def get_critical_recent(limit: int = 5) -> List[RecentActivityItem]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, plant_id, captured_at, health_score, status, main_issue, image_name
            FROM scan_logs
            WHERE status = 'ACT NOW'
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [RecentActivityItem(**dict(r)) for r in rows]
