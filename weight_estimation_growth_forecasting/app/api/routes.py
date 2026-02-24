from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
import os
import base64
from pydantic import ValidationError

from app.core.db_deps import get_db
from app.core.db_models import SensorReading, PlantScan, PredictionLog

from app.schemas import (
    InferRequest,
    InferResponse,
    ForecastRequest,
    ForecastResponse,
    PanelTodayResponse,
    SensorPacket,
    LatestDashboardResponse,
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
    A_proj_cm2, D_proj_cm, Z_m = get_proj_area_and_diam(rgb_bytes, depth_bytes)

    # 3) projected -> leaf area -> weight
    A_des_cm2 = leaf_area_from_proj(A_proj_cm2, D_proj_cm)
    W_today_g = predict_weight_g(A_des_cm2, D_proj_cm)

    # 4) mask overlay for UI
    mask_png = make_mask_overlay_png(rgb_bytes, alpha=0.45)
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    # 5) save instant sensors (optional)
    if payload.sensors is not None:
        db.add(
            SensorReading(
                zone_id=payload.zone_id,
                ts=datetime.utcnow(),
                airT=payload.sensors.airT,
                RH=payload.sensors.RH,
                EC=payload.sensors.EC,
                pH=payload.sensors.pH,
            )
        )
        db.commit()

    # 6) 3-day mean sensors (model inputs)
    means = get_3day_means(db, payload.zone_id, datetime.utcnow()) or {}

    # 7) resolve A_prev from payload or DB (fallback = today A_proj)
    A_prev = payload.A_prev_cm2
    if A_prev is None:
        prev = (
            db.query(PredictionLog)
            .filter(PredictionLog.plant_id == payload.plant_id)
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
        sensors=means,  # dict with mean keys
    )

    # 9) store prediction log (for future history)
    db.add(
        PredictionLog(
            plant_id=payload.plant_id,
            zone_id=payload.zone_id,
            ts=datetime.utcnow(),
            A_proj_cm2=float(A_proj_cm2),
            D_proj_cm=float(D_proj_cm),
            A_leaf_est_cm2=float(A_des_cm2),
            weight_est_g=float(W_today_g),
            A_next_cm2=float(A_tmr),
            D_next_cm=float(D_tmr),
        )
    )
    db.commit()

    # 10) compute tomorrow weight via leaf-area
    A_leaf_tmr = leaf_area_from_proj(A_tmr, D_tmr)
    W_tmr_g = predict_weight_g(A_leaf_tmr, D_tmr)

    return InferResponse(
        A_proj_cm2=A_proj_cm2,
        D_proj_cm=D_proj_cm,
        A_des_cm2=A_des_cm2,
        W_today_g=W_today_g,
        A_proj_tmr_cm2=A_tmr,
        D_proj_tmr_cm=D_tmr,
        W_tmr_g=W_tmr_g,
        mask_overlay_b64=mask_b64,
    )


@router.post("/forecast", response_model=ForecastResponse)
async def infer_forecast(payload: ForecastRequest, db: Session = Depends(get_db)):
    # 1) resolve A_prev from payload or DB (fallback = A_t)
    A_prev = payload.A_prev_cm2
    if A_prev is None:
        prev = (
            db.query(PredictionLog)
            .filter(PredictionLog.plant_id == payload.plant_id)
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

    # 3) enrich each point with leaf area + weight
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

@router.post("/iot/ingest")
def iot_ingest(packet: SensorPacket, db: Session = Depends(get_db)):
    row = SensorReading(**packet.model_dump())
    db.add(row)
    db.commit()
    return {"ok": True}


@router.post("/scans/ingest")
async def scans_ingest(
    device_id: str = Form(...),
    plant_id: str = Form(...),
    zone_id: str = Form(...),
    ts: str = Form(...),
    rgb_image: UploadFile = File(...),
    depth_image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    ts_dt = datetime.fromisoformat(ts)

    rgb_bytes = await rgb_image.read()
    depth_bytes = await depth_image.read() if depth_image else None

    if depth_bytes is None:
        return {"ok": False, "error": "depth_image is required for this vision pipeline"}

    rgb_path = os.path.join(UPLOAD_DIR, f"{plant_id}_{int(ts_dt.timestamp())}_{rgb_image.filename}")
    with open(rgb_path, "wb") as f:
        f.write(rgb_bytes)

    depth_path = os.path.join(UPLOAD_DIR, f"{plant_id}_{int(ts_dt.timestamp())}_{depth_image.filename}")
    with open(depth_path, "wb") as f:
        f.write(depth_bytes)

    db.add(
        PlantScan(
            device_id=device_id,
            plant_id=plant_id,
            zone_id=zone_id,
            ts=ts_dt,
            rgb_path=rgb_path,
            depth_path=depth_path,
        )
    )
    db.commit()

    A_proj_cm2, D_proj_cm, Z_m = get_proj_area_and_diam(rgb_bytes, depth_bytes)

    if A_proj_cm2 <= 0 or D_proj_cm <= 0:
        return {
            "ok": False,
            "error": "vision returned zero area/diameter (bad mask or invalid depth)",
            "A_proj_cm2": float(A_proj_cm2),
            "D_proj_cm": float(D_proj_cm),
            "Z_m": float(Z_m),
        }

    A_leaf_est_cm2 = float(leaf_area_from_proj(A_proj_cm2, D_proj_cm))
    W_g = float(predict_weight_g(A_leaf_est_cm2, D_proj_cm))

    means = get_3day_means(db, zone_id, ts_dt)

    A_next = None
    D_next = None
    if means:
        A_next, D_next = predict_tomorrow(
            dap=25,
            A_t_cm2=A_proj_cm2,
            D_t_cm=D_proj_cm,
            A_prev_cm2=A_proj_cm2,
            sensors=means,
        )
        A_next = float(A_next)
        D_next = float(D_next)

    db.add(
        PredictionLog(
            plant_id=plant_id,
            zone_id=zone_id,
            ts=ts_dt,
            A_proj_cm2=float(A_proj_cm2),
            D_proj_cm=float(D_proj_cm),
            A_leaf_est_cm2=float(A_leaf_est_cm2),
            weight_est_g=float(W_g),
            A_next_cm2=A_next,
            D_next_cm=D_next,
        )
    )
    db.commit()

    return {
        "ok": True,
        "plant_id": plant_id,
        "zone_id": zone_id,
        "ts": ts_dt.isoformat(),
        "A_proj_cm2": float(A_proj_cm2),
        "D_proj_cm": float(D_proj_cm),
        "Z_m": float(Z_m),
        "A_leaf_est_cm2": float(A_leaf_est_cm2),
        "weight_est_g": float(W_g),
        "A_next_cm2": A_next,
        "D_next_cm": D_next,
        "sensor_means_3d": means,
        "saved": {"rgb_path": rgb_path, "depth_path": depth_path},
    }


@router.get("/dashboard/latest", response_model=LatestDashboardResponse)
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