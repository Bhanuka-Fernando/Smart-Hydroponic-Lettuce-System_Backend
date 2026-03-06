from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.schemas import (
    AnalyzeRequest, AnalyzeBatchRequest, AnalyzeResponse,
    IngestRequest, IngestResponse, LatestResponse, HistoryResponse
)
from app.core.config import settings
from app.core.db_deps import get_db
from app.core.db_models import WaterReading
from app.services.predictors import DualPredictor
from app.services.features import build_feature_frame, latest_feature_row, get_turb_delta_30min
from app.services.rules import water_rule_checks, algae_reasoning, health_score_from_severity
from app.services.postprocess import confidence_gate_ml, worst_status, sensor_quality_checks

router = APIRouter(prefix="/water", tags=["water"])

_predictor: Optional[DualPredictor] = None

def get_predictor() -> DualPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DualPredictor(settings.water_model_path, settings.algae_model_path)
    return _predictor

@router.get("/health")
def health():
    return {"status": "ok", "service": settings.service_name}

def parse_ts(ts: str) -> datetime:
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(dt):
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    return dt.to_pydatetime()

@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, db: Session = Depends(get_db)):
    if not req.readings:
        raise HTTPException(status_code=400, detail="readings[] is empty")

    saved = 0
    for r in req.readings:
        dt = parse_ts(r.timestamp)
        row = WaterReading(
            tank_id=req.tank_id.strip(),
            timestamp=dt,
            ph=float(r.ph),
            temp_c=float(r.temp_c),
            turb_ntu=float(r.turb_ntu),
            ec=float(r.ec),
        )
        db.add(row)
        saved += 1

    db.commit()
    return IngestResponse(saved=saved, tank_id=req.tank_id.strip())

@router.get("/latest", response_model=LatestResponse)
def latest(tank_id: str = Query(...), db: Session = Depends(get_db)):
    row = (
        db.query(WaterReading)
        .filter(WaterReading.tank_id == tank_id)
        .order_by(WaterReading.timestamp.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="No readings found for tank_id")

    return LatestResponse(
        tank_id=row.tank_id,
        timestamp=row.timestamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        ph=row.ph,
        temp_c=row.temp_c,
        turb_ntu=row.turb_ntu,
        ec=row.ec,
    )

@router.get("/history", response_model=HistoryResponse)
def history(
    tank_id: str = Query(...),
    limit: int = Query(60, ge=1, le=2000),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(WaterReading)
        .filter(WaterReading.tank_id == tank_id)
        .order_by(WaterReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    rows = list(reversed(rows))
    out = [
        LatestResponse(
            tank_id=r.tank_id,
            timestamp=r.timestamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            ph=r.ph,
            temp_c=r.temp_c,
            turb_ntu=r.turb_ntu,
            ec=r.ec,
        )
        for r in rows
    ]
    return HistoryResponse(tank_id=tank_id, count=len(out), readings=out)

def rows_from_db(tank_id: str, db: Session, minutes: int) -> List[tuple]:
    """
    Fetch last N minutes history from DB for single-reading analysis.
    """
    now = datetime.now(timezone.utc)
    start = now - pd.Timedelta(minutes=minutes)
    rows = (
        db.query(WaterReading)
        .filter(WaterReading.tank_id == tank_id)
        .filter(WaterReading.timestamp >= start.to_pydatetime())
        .order_by(WaterReading.timestamp.asc())
        .all()
    )
    return [
        (r.timestamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"), r.ph, r.temp_c, r.turb_ntu, r.ec)
        for r in rows
    ]

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    predictor = get_predictor()
    tank_id = req.tank_id.strip()

    # Save the incoming reading to DB (so single mode can have history)
    dt = parse_ts(req.timestamp)
    db.add(WaterReading(
        tank_id=tank_id,
        timestamp=dt,
        ph=float(req.ph),
        temp_c=float(req.temp_c),
        turb_ntu=float(req.turb_ntu),
        ec=float(req.ec),
    ))
    db.commit()

    # Pull history from DB for rolling features
    hist_rows = rows_from_db(tank_id, db, minutes=settings.history_minutes)
    if len(hist_rows) < 4:
        raise HTTPException(status_code=400, detail="Not enough history yet. Send more readings or use /analyze_batch.")

    df_feat = build_feature_frame(hist_rows, settings.resample_rule)
    x = latest_feature_row(df_feat, predictor.feature_cols)
    if x is None:
        raise HTTPException(status_code=400, detail="Not enough resampled points for rolling features (need ~1 hour).")

    ml_status, ml_probs, ml_algae, ml_algae_probs = predictor.predict(x)

    turb_d2 = get_turb_delta_30min(df_feat)

    sensor_quality, sensor_notes = sensor_quality_checks(req.ph, req.temp_c, req.turb_ntu, req.ec)

    rule_status, sev, reasons, actions = water_rule_checks(req.ph, req.temp_c, req.turb_ntu, req.ec, turb_d2)
    health_score = health_score_from_severity(sev, reasons)
    score_status = rule_status

    ml_status_gated = confidence_gate_ml(ml_status, ml_probs, sensor_quality)
    final_status = worst_status(rule_status, ml_status_gated)

    if final_status != rule_status and len(reasons) == 0:
        reasons = ["Detected risky pattern from recent sensor trends"]
        actions = ["Recheck sensors and inspect the system"]

    algae_reasons, algae_actions = algae_reasoning(req.turb_ntu, turb_d2, req.temp_c, req.ec, req.ph)

    return AnalyzeResponse(
        tank_id=tank_id,
        timestamp=req.timestamp,
        mode="single",
        ml_status=ml_status,
        ml_probs=ml_probs,
        ml_algae=ml_algae,
        ml_algae_probs=ml_algae_probs,
        rule_status=rule_status,
        health_score=health_score,
        score_status=score_status,
        final_status=final_status,
        reasons=reasons,
        actions=actions,
        algae_reasons=algae_reasons,
        algae_actions=algae_actions,
        sensor_quality=sensor_quality,
        sensor_notes=sensor_notes,
        meta={
            "history_minutes": settings.history_minutes,
            "resample_rule": settings.resample_rule,
            "resampled_points": int(len(df_feat)),
            "turb_delta_30min": turb_d2,
        },
    )

@router.post("/analyze_batch", response_model=AnalyzeResponse)
def analyze_batch(req: AnalyzeBatchRequest):
    predictor = get_predictor()
    tank_id = req.tank_id.strip()

    if not req.readings:
        raise HTTPException(status_code=400, detail="readings[] is empty")

    rows = [(r.timestamp, r.ph, r.temp_c, r.turb_ntu, r.ec) for r in req.readings]
    df_feat = build_feature_frame(rows, settings.resample_rule)
    x = latest_feature_row(df_feat, predictor.feature_cols)
    if x is None:
        raise HTTPException(status_code=400, detail="Not enough readings for rolling features. Provide at least ~1 hour of data.")

    ml_status, ml_probs, ml_algae, ml_algae_probs = predictor.predict(x)

    last_raw = req.readings[-1]
    turb_d2 = get_turb_delta_30min(df_feat)

    sensor_quality, sensor_notes = sensor_quality_checks(last_raw.ph, last_raw.temp_c, last_raw.turb_ntu, last_raw.ec)

    rule_status, sev, reasons, actions = water_rule_checks(last_raw.ph, last_raw.temp_c, last_raw.turb_ntu, last_raw.ec, turb_d2)
    health_score = health_score_from_severity(sev, reasons)
    score_status = rule_status

    ml_status_gated = confidence_gate_ml(ml_status, ml_probs, sensor_quality)
    final_status = worst_status(rule_status, ml_status_gated)

    if final_status != rule_status and len(reasons) == 0:
        reasons = ["Detected risky pattern from recent sensor trends"]
        actions = ["Recheck sensors and inspect the system"]

    algae_reasons, algae_actions = algae_reasoning(last_raw.turb_ntu, turb_d2, last_raw.temp_c, last_raw.ec, last_raw.ph)

    # Use latest resampled timestamp
    latest_time = df_feat["timestamp"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    return AnalyzeResponse(
        tank_id=tank_id,
        timestamp=latest_time,
        mode="batch_timeseries",
        ml_status=ml_status,
        ml_probs=ml_probs,
        ml_algae=ml_algae,
        ml_algae_probs=ml_algae_probs,
        rule_status=rule_status,
        health_score=health_score,
        score_status=score_status,
        final_status=final_status,
        reasons=reasons,
        actions=actions,
        algae_reasons=algae_reasons,
        algae_actions=algae_actions,
        sensor_quality=sensor_quality,
        sensor_notes=sensor_notes,
        meta={
            "resample_rule": settings.resample_rule,
            "resampled_points": int(len(df_feat)),
            "turb_delta_30min": turb_d2,
        },
    )