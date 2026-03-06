from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
import os
import base64
from typing import Optional, List
from pydantic import ValidationError
from app.schemas import GrowthPredictSaveRequest
from app.core.db_models import GrowthPredictionLog, PlantMeta

from app.core.db_deps import get_db
from app.core.db_models import SensorReading, PlantScan, PredictionLog, Activity

from app.schemas import (
    InferRequest,
    InferResponse,
    ForecastRequest,
    ForecastResponse,
    PanelTodayResponse,
    SensorPacket,
    LatestDashboardResponse,
    WeightSaveRequest,
    PlantDetailsResponse,
    PlantHistoryItem,
    DashboardMetricsResponse,
    IoTSensorPayload,
    IoTIngestResponse,
    ActivityItem,
    ActivityHistoryResponse,
)

from app.services.vision import (
    get_proj_area_and_diam,
    make_mask_applied_png,
    make_mask_overlay_png,
)
from app.services.leaf_area import leaf_area_from_proj
from app.services.weight import predict_weight_g
from app.services.growth import predict_tomorrow, forecast_n_days
from app.services.iot_agg import get_3day_means

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/infer", tags=["inference"])


def _date_label(ts: datetime) -> str:
    today = datetime.utcnow().date()
    d = ts.date()
    if d == today:
        return f"Today, {ts.strftime('%b %d')}"
    return ts.strftime("%b %d")


@router.post("/today", response_model=InferResponse)
async def infer_today(
    payload_json: str = Form(...),
    image: UploadFile = File(...),
    depth: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 0) validate payload json
    try:
        payload = InferRequest.model_validate_json(payload_json)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    # 1) read image bytes
    rgb_bytes = await image.read()
    depth_bytes = await depth.read()

    # 2) vision extractor
    A_proj_cm2, D_proj_cm, _ = get_proj_area_and_diam(rgb_bytes, depth_bytes)

    # 3) projected -> leaf area -> weight
    A_des_cm2 = leaf_area_from_proj(A_proj_cm2, D_proj_cm)
    W_today_g = predict_weight_g(A_des_cm2, D_proj_cm)

    # 4) mask overlay for UI
    mask_png = make_mask_overlay_png(rgb_bytes, alpha=0.45)
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")
    
    # 4.5) ✅ SAVE UPLOADED IMAGES TO DISK
    now = datetime.utcnow()  # Single timestamp for all operations
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    rgb_filename = f"{payload.plant_id}_{timestamp_str}_rgb.png"
    depth_filename = f"{payload.plant_id}_{timestamp_str}_depth.png"
    
    rgb_path = os.path.join(UPLOAD_DIR, rgb_filename)
    depth_path = os.path.join(UPLOAD_DIR, depth_filename)
    
    with open(rgb_path, "wb") as f:
        f.write(rgb_bytes)
    with open(depth_path, "wb") as f:
        f.write(depth_bytes)

    # 5) save instant sensors (optional)
    if payload.sensors is not None:
        db.add(
            SensorReading(
                zone_id=payload.zone_id,
                ts=now,
                airT=payload.sensors.airT,
                RH=payload.sensors.RH,
                EC=payload.sensors.EC,
                pH=payload.sensors.pH,
            )
        )
        db.commit()

    # 6) 3-day mean sensors (model inputs)
    means = get_3day_means(db, payload.zone_id, now) or {}

    # 7) resolve A_prev from payload or DB (fallback = today A_proj)
    A_prev = payload.A_prev_cm2
    if A_prev is None:
        prev = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.plant_id == payload.plant_id,
                PredictionLog.zone_id == payload.zone_id,
            )
            .order_by(PredictionLog.ts.desc())
            .first()
        )
        A_prev = float(prev.A_proj_cm2) if prev and prev.A_proj_cm2 is not None else float(A_proj_cm2)

    # 8) tomorrow prediction (projected A + D)
    A_tmr, D_tmr = predict_tomorrow(
        dap=payload.dap,
        A_t_cm2=A_proj_cm2,
        D_t_cm=D_proj_cm,
        A_prev_cm2=A_prev,
        sensors=means,
    )

    # 10) compute tomorrow weight
    A_leaf_tmr = leaf_area_from_proj(A_tmr, D_tmr)
    W_tmr_g = predict_weight_g(A_leaf_tmr, D_tmr)

    # 11) ✅ SAVE SCAN TO DATABASE (auto-persist every scan)
    # Check for duplicate (same plant, zone, timestamp within 1 minute)
    existing = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.plant_id == payload.plant_id,
            PredictionLog.zone_id == payload.zone_id,
            PredictionLog.ts >= now - timedelta(minutes=1),
            PredictionLog.ts <= now + timedelta(minutes=1),
        )
        .first()
    )
    
    if not existing:
        # Save scan to PredictionLog
        scan_log = PredictionLog(
            plant_id=payload.plant_id,
            zone_id=payload.zone_id,
            ts=now,
            A_proj_cm2=float(A_proj_cm2),
            D_proj_cm=float(D_proj_cm),
            A_leaf_est_cm2=float(A_des_cm2),
            weight_est_g=float(W_today_g),
            A_next_cm2=float(A_tmr),
            D_next_cm=float(D_tmr),
        )
        db.add(scan_log)
        
        # Save PlantScan (image references)
        plant_scan = PlantScan(
            device_id="mobile-app",
            plant_id=payload.plant_id,
            zone_id=payload.zone_id,
            ts=now,
            rgb_path=rgb_path,
            depth_path=depth_path,
        )
        db.add(plant_scan)
        
        # Create or update PlantMeta (for age and weight tracking)
        meta = (
            db.query(PlantMeta)
            .filter(
                PlantMeta.plant_id == payload.plant_id,
                PlantMeta.zone_id == payload.zone_id,
            )
            .first()
        )
        
        if not meta:
            # Calculate planted_at from DAP (days after planting)
            planted_at = now - timedelta(days=payload.dap)
            meta = PlantMeta(
                plant_id=payload.plant_id,
                zone_id=payload.zone_id,
                planted_at=planted_at,
                updated_at=now,
                start_weight_g=float(W_today_g),  # ✅ Set once on first scan
                current_weight_g=float(W_today_g),
            )
            db.add(meta)
        else:
            # ✅ Only set start_weight_g if NULL (first scan after growth prediction)
            if meta.start_weight_g is None:
                meta.start_weight_g = float(W_today_g)
            # ✅ Always update current_weight_g with latest scan
            meta.current_weight_g = float(W_today_g)
            meta.updated_at = now
        
        db.commit()

    return InferResponse(
        A_proj_cm2=A_proj_cm2,
        D_proj_cm=D_proj_cm,
        A_des_cm2=A_des_cm2,
        W_today_g=W_today_g,
        A_proj_tmr_cm2=A_tmr,
        D_proj_tmr_cm=D_tmr,
        W_tmr_g=W_tmr_g,
        mask_overlay_b64=mask_b64,
        image_url=rgb_path if 'rgb_path' in locals() else None,
        captured_at=now.isoformat() if 'now' in locals() else None,
        plant_id=payload.plant_id,
        zone_id=payload.zone_id,
    )


@router.post("/forecast", response_model=ForecastResponse)
async def infer_forecast(payload: ForecastRequest, db: Session = Depends(get_db)):
    # 1) resolve A_prev from payload or DB (fallback = A_t)
    A_prev = payload.A_prev_cm2
    if A_prev is None:
        prev = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.plant_id == payload.plant_id,
                PredictionLog.zone_id == payload.zone_id,
            )
            .order_by(PredictionLog.ts.desc())
            .first()
        )
        A_prev = float(prev.A_proj_cm2) if prev and prev.A_proj_cm2 is not None else float(payload.A_t_cm2)

    # 2) get raw projected forecasts
    points_raw = forecast_n_days(
        dap_start=payload.dap,
        A_prev_cm2=A_prev,
        A_t_cm2=payload.A_t_cm2,
        D_t_cm=payload.D_t_cm,
        sensors=payload.sensors,
        n_days=payload.n_days,
    )

    # 3) enrich
    points_out = []
    for p in points_raw:
        A_proj = float(p["A_pred_cm2"])
        D_cm = float(p["D_pred_cm"])

        A_leaf = float(leaf_area_from_proj(A_proj, D_cm))
        W_g = float(predict_weight_g(A_leaf, D_cm))

        points_out.append(
            {
                "step": int(p["step"]),
                "DAP_pred": int(p["DAP_pred"]),
                "A_pred_cm2": A_proj,
                "D_pred_cm": D_cm,
                "A_leaf_pred_cm2": A_leaf,
                "W_pred_g": W_g,
            }
        )

    # Log activity
    activity = Activity(
        activity_type="growth_forecast",
        title=f"Growth forecast for {payload.plant_id}",
        description=f"Forecast for {payload.n_days} days starting from DAP {payload.dap}",
        zone_id=payload.zone_id,
        status="success",
    )
    db.add(activity)
    db.commit()

    return ForecastResponse(points=points_out)


@router.post("/panel/today", response_model=PanelTodayResponse)
async def panel_today(
    payload_json: str = Form(...),
    image: UploadFile = File(...),
    depth: UploadFile = File(...),
):
    payload = InferRequest.model_validate_json(payload_json)

    rgb_bytes = await image.read()
    depth_bytes = await depth.read()

    A_proj_cm2, D_cm, _ = get_proj_area_and_diam(rgb_bytes, depth_bytes)
    A_leaf_cm2 = leaf_area_from_proj(A_proj_cm2, D_cm)
    W_today_g = predict_weight_g(A_leaf_cm2, D_cm)

    A_prev = payload.A_prev_cm2 if payload.A_prev_cm2 is not None else A_proj_cm2
    A_proj_tmr, D_tmr = predict_tomorrow(payload.dap, A_proj_cm2, D_cm, A_prev, payload.sensors)

    A_leaf_tmr_cm2 = leaf_area_from_proj(A_proj_tmr, D_tmr)
    W_tmr_g = predict_weight_g(A_leaf_tmr_cm2, D_tmr)

    return PanelTodayResponse(
        Leaf_Area_today_cm2=A_leaf_cm2,
        Diameter_today_cm=D_cm,
        Weight_today_g=W_today_g,
        Leaf_Area_tomorrow_cm2=A_leaf_tmr_cm2,
        Diameter_tomorrow_cm=D_tmr,
        Weight_tomorrow_g=W_tmr_g,
    )


@router.post("/mask/applied")
async def download_mask_applied(image: UploadFile = File(...)):
    rgb_bytes = await image.read()
    png_bytes = make_mask_applied_png(rgb_bytes)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="plant_mask_applied.png"'},
    )


@router.post("/mask/overlay")
async def download_mask_overlay(image: UploadFile = File(...)):
    rgb_bytes = await image.read()
    png_bytes = make_mask_overlay_png(rgb_bytes, alpha=0.5)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="plant_mask_overlay.png"'},
    )


# -------------------------
# IOT ROUTES
# -------------------------

@router.post("/iot/ingest", response_model=IoTIngestResponse)
def iot_ingest_new(payload: IoTSensorPayload, db: Session = Depends(get_db)):
    """Frontend-compatible IoT sensor data ingestion"""
    # Parse timestamp or use current time
    if payload.timestamp:
        try:
            from dateutil import parser
            ts = parser.isoparse(payload.timestamp)
        except:
            ts = datetime.utcnow()
    else:
        ts = datetime.utcnow()
    
    # Validate sensor ranges
    if not (15 <= payload.temperature_c <= 35):
        raise HTTPException(status_code=400, detail="Temperature must be between 15-35°C")
    if not (30 <= payload.humidity_pct <= 90):
        raise HTTPException(status_code=400, detail="Humidity must be between 30-90%")
    if not (0.5 <= payload.ec_ms_cm <= 3.0):
        raise HTTPException(status_code=400, detail="EC must be between 0.5-3.0 mS/cm")
    if not (4.0 <= payload.ph <= 8.0):
        raise HTTPException(status_code=400, detail="pH must be between 4.0-8.0")
    
    row = SensorReading(
        zone_id=payload.zone_id,
        ts=ts,
        airT=payload.temperature_c,
        RH=payload.humidity_pct,
        EC=payload.ec_ms_cm,
        pH=payload.ph,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    
    # Log activity
    activity = Activity(
        activity_type="sensor_update",
        title=f"Sensor data recorded for {payload.zone_id}",
        description=f"T: {payload.temperature_c}°C, RH: {payload.humidity_pct}%, EC: {payload.ec_ms_cm}, pH: {payload.ph}",
        zone_id=payload.zone_id,
        status="success",
    )
    db.add(activity)
    db.commit()
    
    return IoTIngestResponse(
        ok=True,
        sensor_id=str(row.id),
        recorded_at=row.ts.isoformat(),
    )


# Legacy endpoint for backward compatibility
@router.post("/iot/ingest/legacy")
def iot_ingest(packet: SensorPacket, db: Session = Depends(get_db)):
    row = SensorReading(**packet.model_dump())
    db.add(row)
    db.commit()
    return {"ok": True}


@router.get("/dashboard/latest", response_model=DashboardMetricsResponse)
def dashboard_latest_new(zone_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Frontend-compatible dashboard metrics endpoint"""
    # Get latest sensor reading
    sensor_query = db.query(SensorReading)
    if zone_id:
        sensor_query = sensor_query.filter(SensorReading.zone_id == zone_id)
    latest_sensor = sensor_query.order_by(desc(SensorReading.ts)).first()
    
    # Count total plants in zone
    plant_query = db.query(PredictionLog.plant_id).distinct()
    if zone_id:
        plant_query = plant_query.filter(PredictionLog.zone_id == zone_id)
    plant_count = plant_query.count()
    
    # Count harvest-ready plants (>= 300g)
    harvest_query = db.query(PredictionLog).filter(PredictionLog.weight_est_g >= 300)
    if zone_id:
        harvest_query = harvest_query.filter(PredictionLog.zone_id == zone_id)
    
    # Get latest weight for each plant
    subquery = db.query(
        PredictionLog.plant_id,
        PredictionLog.weight_est_g
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
                PredictionLog.plant_id == plant_id
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


# Legacy dashboard endpoint
@router.get("/dashboard/latest/legacy", response_model=LatestDashboardResponse)
def dashboard_latest(zone_id: str, plant_id: str, db: Session = Depends(get_db)):
    latest_sensor = (
        db.query(SensorReading)
        .filter(SensorReading.zone_id == zone_id)
        .order_by(desc(SensorReading.ts))
        .first()
    )

    latest_pred = (
        db.query(PredictionLog)
        .filter(PredictionLog.zone_id == zone_id, PredictionLog.plant_id == plant_id)
        .order_by(desc(PredictionLog.ts))
        .first()
    )

    now = latest_pred.ts if latest_pred else (latest_sensor.ts if latest_sensor else datetime.utcnow())
    means = get_3day_means(db, zone_id, now) or {}

    return LatestDashboardResponse(
        zone_id=zone_id,
        plant_id=plant_id,
        ts=now,
        airT=getattr(latest_sensor, "airT", None) if latest_sensor else None,
        RH=getattr(latest_sensor, "RH", None) if latest_sensor else None,
        EC=getattr(latest_sensor, "EC", None) if latest_sensor else None,
        pH=getattr(latest_sensor, "pH", None) if latest_sensor else None,
        **means,
        A_proj_cm2=getattr(latest_pred, "A_proj_cm2", None) if latest_pred else None,
        D_proj_cm=getattr(latest_pred, "D_proj_cm", None) if latest_pred else None,
        A_leaf_est_cm2=getattr(latest_pred, "A_leaf_est_cm2", None) if latest_pred else None,
        weight_est_g=getattr(latest_pred, "weight_est_g", None) if latest_pred else None,
        A_next_cm2=getattr(latest_pred, "A_next_cm2", None) if latest_pred else None,
        D_next_cm=getattr(latest_pred, "D_next_cm", None) if latest_pred else None,
    )


@router.post("/weights/save")
def save_weight_result(payload: WeightSaveRequest, db: Session = Depends(get_db)):
    existing = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.plant_id == payload.plant_id,
            PredictionLog.zone_id == payload.zone_id,
            PredictionLog.ts == payload.captured_at,
        )
        .first()
    )
    if existing:
        return {"ok": True, "deduped": True}

    row = PredictionLog(
        plant_id=payload.plant_id,
        zone_id=payload.zone_id,
        ts=payload.captured_at,
        A_proj_cm2=float(payload.A_proj_cm2),
        D_proj_cm=float(payload.D_proj_cm),
        A_leaf_est_cm2=float(payload.A_des_cm2),
        weight_est_g=float(payload.W_today_g),
        A_next_cm2=None,
        D_next_cm=None,
    )
    db.add(row)
    db.commit()
    return {"ok": True}

@router.get("/plants/{plant_id}", response_model=PlantDetailsResponse)
def get_plant_details(
    plant_id: str,
    zone_id: str | None = None,
    range: str = Query("7d", pattern="^(7d|month|all)$"),
    db: Session = Depends(get_db),
):
    # ---- meta (age + planted_on) ----
    meta_q = db.query(PlantMeta).filter(PlantMeta.plant_id == plant_id)
    if zone_id:
        meta_q = meta_q.filter(PlantMeta.zone_id == zone_id)
    meta = meta_q.first()

    planted_at = meta.planted_at if meta else None
    if planted_at:
        age_days = max(0, (datetime.utcnow().date() - planted_at.date()).days)
        planted_on = f"Planted {planted_at.strftime('%b %d')}"
    else:
        age_days = 0
        planted_on = "Planted --"

    # ---- logs (real scans) - GET ALL SCANS (not filtered by range) ----
    q = db.query(PredictionLog).filter(
        PredictionLog.plant_id == plant_id,
        PredictionLog.deleted_at.is_(None)
    )
    if zone_id:
        q = q.filter(PredictionLog.zone_id == zone_id)

    # Get ALL scans for complete history
    logs = q.order_by(PredictionLog.ts.asc()).all()

    # ✅ if no scans yet, fallback to GrowthPredictionLog (prediction-only plants)
    if not logs:
        pred_q = db.query(GrowthPredictionLog).filter(GrowthPredictionLog.plant_id == plant_id)
        pred = pred_q.order_by(desc(GrowthPredictionLog.id)).first()

        if not pred and not meta:
            raise HTTPException(status_code=404, detail="No records for this plant_id")

        # ✅ Use stored weights from PlantMeta if available, else use prediction
        if meta and meta.start_weight_g is not None and meta.current_weight_g is not None:
            start_w = float(meta.start_weight_g)
            current_w = float(meta.current_weight_g)
        else:
            current_w = float(getattr(pred, "predicted_weight_g", 0.0) or 0.0) if pred else 0.0
            start_w = current_w  # Start weight = current for prediction-only plants
        
        growth_pct = ((current_w - start_w) / start_w * 100.0) if start_w > 0 else 0.0

        history = []
        if pred:
            history = [
                PlantHistoryItem(
                    date=datetime.utcnow().date().isoformat(),
                    date_label=_date_label(datetime.utcnow()),
                    actual_weight_g=None,
                    predicted_weight_g=float(getattr(pred, "predicted_weight_g", 0.0) or 0.0),
                    age_days=age_days,
                    delta_g=None,
                    status="Predicted",
                )
            ]

        return PlantDetailsResponse(
            plant_id=plant_id,
            display_name=f"Plant {plant_id}",
            planted_on=planted_on,
            age_days=age_days,
            start_weight_g=round(start_w, 2),
            current_weight_g=round(current_w, 2),
            growth_pct=round(growth_pct, 2),
            predicted_today_g=round(current_w, 2) if pred else None,
            trajectory={"labels": [], "values": []},
            history=history,
        )

    # ---- normal path: we have scans ----
    oldest = logs[0]
    latest = logs[-1]

    # if no PlantMeta, derive planted_at from first scan
    if not planted_at:
        planted_at = oldest.ts
        age_days = max(0, (datetime.utcnow().date() - planted_at.date()).days)
        planted_on = f"Planted {planted_at.strftime('%b %d')}"

    # ✅ Use stored weights from PlantMeta (set once on first scan, never changes)
    # Fallback to calculated values for backward compatibility
    if meta and meta.start_weight_g is not None:
        start_w = float(meta.start_weight_g)
    else:
        start_w = float(oldest.weight_est_g or 0.0)
    
    if meta and meta.current_weight_g is not None:
        current_w = float(meta.current_weight_g)
    else:
        current_w = float(latest.weight_est_g or 0.0)
    
    growth_pct = ((current_w - start_w) / start_w * 100.0) if start_w > 0 else 0.0

    labels = [l.ts.strftime("%b %d") for l in logs]
    values = [float(l.weight_est_g or 0.0) for l in logs]

    # ✅ history - ALL scans with age_days and proper status
    history_forward: list[PlantHistoryItem] = []
    prev_actual = None
    for l in logs:
        actual = float(l.weight_est_g) if l.weight_est_g is not None else None
        delta = (actual - prev_actual) if (actual is not None and prev_actual is not None) else None
        
        # Calculate age_days for this scan
        scan_age_days = max(0, (l.ts.date() - planted_at.date()).days) if planted_at else None
        
        history_forward.append(
            PlantHistoryItem(
                date=l.ts.date().isoformat(),
                date_label=_date_label(l.ts),
                actual_weight_g=round(actual, 2) if actual is not None else None,
                predicted_weight_g=round(actual, 2) if actual is not None else None,  # For scans, predicted = actual
                age_days=scan_age_days,
                delta_g=round(delta, 2) if delta is not None else None,
                status="Scanned",
            )
        )
        if actual is not None:
            prev_actual = actual

    return PlantDetailsResponse(
        plant_id=plant_id,
        display_name=f"Plant {plant_id}",
        planted_on=planted_on,
        age_days=age_days,
        start_weight_g=round(start_w, 2),
        current_weight_g=round(current_w, 2),
        growth_pct=round(growth_pct, 2),
        predicted_today_g=None,
        trajectory={"labels": labels, "values": [round(v, 2) for v in values]},
        history=list(reversed(history_forward)),
    )

# -------------------------
# ACTIVITIES HISTORY
# -------------------------

@router.get("/activities/history", response_model=ActivityHistoryResponse)
def get_activities_history(
    zone_id: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Get activity history with filtering and pagination"""
    query = db.query(Activity)
    
    if zone_id:
        query = query.filter(Activity.zone_id == zone_id)
    
    if type:
        query = query.filter(Activity.activity_type == type)
    
    total_count = query.count()
    
    activities_db = query.order_by(desc(Activity.timestamp)).offset(offset).limit(limit).all()
    
    activities = [
        ActivityItem(
            id=str(a.id),
            type=a.activity_type,
            title=a.title,
            description=a.description or "",
            timestamp=a.timestamp.isoformat(),
            zone=a.zone_id,
            status=a.status,
        )
        for a in activities_db
    ]
    
    has_more = (offset + limit) < total_count
    
    return ActivityHistoryResponse(
        activities=activities,
        total_count=total_count,
        has_more=has_more,
    )