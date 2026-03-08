from datetime import datetime, timezone, timedelta
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.services.sim_manager import ProbReplaySimulator
from app.db import get_session
from app.models import SpoilagePrediction, SpoilageAlert
from app.core.security import require_user
from app.core.config import settings

from app.schemas import (
    SpoilagePredictResponse,
    StageOnlyResponse,
    RemainingDaysOnlyRequest,
    RemainingDaysOnlyResponse,
    StageProbs,
)

from app.services.spoilage_classifier import SpoilageClassifier
from app.services.remaining_days import RemainingDaysRegressor
from app.services.postprocess import make_status
from app.services.plant_id import normalize_plant_id

router = APIRouter()

clf = SpoilageClassifier(settings.STAGE_MODEL_PATH, settings.STAGE_META_PATH)
reg = RemainingDaysRegressor(settings.REG_MODEL_PATH, settings.REG_META_PATH)
sim = ProbReplaySimulator(settings.SIM_PROBS_CSV)

STAGE_ORDER = ["fresh", "slightly_aged", "near_spoilage", "spoiled"]
STAGE_RANK = {name: i for i, name in enumerate(STAGE_ORDER)}


def _normalize_stage(stage: str | None) -> str | None:
    if not stage:
        return None
    s = stage.strip().lower()
    return s if s in STAGE_RANK else None


def _parse_dt_or_now(value: str | None) -> datetime:
    if not value or value.strip().lower() in ("string", "null", "none"):
        return datetime.now(timezone.utc)

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _cap_remaining_by_stage(stage: str, remaining: float) -> float:
    s = _normalize_stage(stage)
    value = max(0.0, float(remaining))

    if s == "spoiled":
        return 0.0
    if s == "near_spoilage":
        return min(value, 2.0)
    if s == "slightly_aged":
        return min(value, 5.0)

    return value


def _alert_payload_for_stage(stage: str, plant_id: str) -> dict | None:
    if stage == "near_spoilage":
        return {
            "severity": "warning",
            "title": "Near spoilage detected",
            "message": f"Plant {plant_id} has entered near spoilage stage. Inspect and take action soon.",
        }

    if stage == "spoiled":
        return {
            "severity": "critical",
            "title": "Spoilage detected",
            "message": f"Plant {plant_id} is now spoiled or critical. Immediate attention is needed.",
        }

    return None


def predict_current_state(
    img_bytes: bytes,
    temperature: float,
    humidity: float,
) -> dict:
    raw_stage, probs = clf.predict(img_bytes, temperature, humidity)
    raw_remaining = reg.predict(probs, temperature, humidity)

    final_stage = raw_stage
    final_remaining = _cap_remaining_by_stage(final_stage, raw_remaining)
    final_status = make_status(final_stage, probs)

    return {
        "final_stage": final_stage,
        "probs": probs,
        "final_remaining": float(final_remaining),
        "final_status": final_status,
    }


def load_latest_prediction(
    session: Session,
    plant_id: str,
) -> SpoilagePrediction | None:
    stmt = (
        select(SpoilagePrediction)
        .where(SpoilagePrediction.plant_id == plant_id)
        .order_by(SpoilagePrediction.captured_at.desc())
        .limit(1)
    )
    return session.exec(stmt).first()


def save_prediction_row(
    session: Session,
    latest: SpoilagePrediction | None,
    plant_id: str,
    captured_dt: datetime,
    temperature: float,
    humidity: float,
    final_stage: str,
    final_status: str,
    final_remaining: float,
    probs: dict,
    image_url: str,
    sim_source_image: str | None,
) -> SpoilagePrediction:
    if latest:
        latest_dt = latest.captured_at
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)

        diff_sec = abs((captured_dt - latest_dt).total_seconds())

        if diff_sec <= 60:
            latest.captured_at = captured_dt
            latest.temperature = float(temperature)
            latest.humidity = float(humidity)
            latest.stage = final_stage
            latest.status = final_status
            latest.remaining_days = float(final_remaining)
            latest.p_fresh = float(probs["fresh"])
            latest.p_slightly_aged = float(probs["slightly_aged"])
            latest.p_near_spoilage = float(probs["near_spoilage"])
            latest.p_spoiled = float(probs["spoiled"])
            latest.image_url = image_url
            latest.sim_source_image = sim_source_image

            session.add(latest)
            session.commit()
            session.refresh(latest)
            return latest

    row = SpoilagePrediction(
        plant_id=plant_id,
        captured_at=captured_dt,
        temperature=float(temperature),
        humidity=float(humidity),
        stage=final_stage,
        status=final_status,
        remaining_days=float(final_remaining),
        p_fresh=float(probs["fresh"]),
        p_slightly_aged=float(probs["slightly_aged"]),
        p_near_spoilage=float(probs["near_spoilage"]),
        p_spoiled=float(probs["spoiled"]),
        image_url=image_url,
        sim_source_image=sim_source_image,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _create_alert_if_needed(
    session: Session,
    plant_id: str,
    prediction_id: int | None,
    previous_stage: str | None,
    current_stage: str,
):
    if current_stage not in ("near_spoilage", "spoiled"):
        return None

    if previous_stage == current_stage:
        return None

    payload = _alert_payload_for_stage(current_stage, plant_id)
    if not payload:
        return None

    recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)

    recent_stmt = (
        select(SpoilageAlert)
        .where(SpoilageAlert.plant_id == plant_id)
        .where(SpoilageAlert.stage == current_stage)
        .where(SpoilageAlert.created_at >= recent_cutoff)
        .order_by(SpoilageAlert.created_at.desc())
        .limit(1)
    )
    recent = session.exec(recent_stmt).first()
    if recent:
        return None

    alert = SpoilageAlert(
        plant_id=plant_id,
        prediction_id=prediction_id,
        stage=current_stage,
        severity=payload["severity"],
        title=payload["title"],
        message=payload["message"],
    )
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def _latest_prediction_rows(session: Session, limit: int = 200) -> list[SpoilagePrediction]:
    rows = session.exec(
        select(SpoilagePrediction)
        .order_by(SpoilagePrediction.captured_at.desc())
        .limit(limit)
    ).all()

    seen = set()
    latest_rows: list[SpoilagePrediction] = []
    for r in rows:
        if r.plant_id not in seen:
            seen.add(r.plant_id)
            latest_rows.append(r)
    return latest_rows


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/spoilage/predict", response_model=SpoilagePredictResponse)
async def spoilage_predict(
    user=Depends(require_user),
    session: Session = Depends(get_session),
    image: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    plant_id: str = Form(...),
    captured_at: str | None = Form(None),
    sim_source_image: str | None = Form(None),
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        plant_id = normalize_plant_id(plant_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    captured_dt = _parse_dt_or_now(captured_at)

    Path("uploads").mkdir(exist_ok=True)
    filename = f"{plant_id}_{uuid.uuid4().hex}.jpg"
    file_path = Path("uploads") / filename
    file_path.write_bytes(img_bytes)
    image_url = f"/uploads/{filename}"

    try:
        prediction = predict_current_state(img_bytes, temperature, humidity)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    final_stage = prediction["final_stage"]
    probs = prediction["probs"]
    final_remaining = prediction["final_remaining"]
    final_status = prediction["final_status"]

    latest = load_latest_prediction(session, plant_id)
    previous_stage = latest.stage if latest else None

    try:
        saved_row = save_prediction_row(
            session=session,
            latest=latest,
            plant_id=plant_id,
            captured_dt=captured_dt,
            temperature=temperature,
            humidity=humidity,
            final_stage=final_stage,
            final_status=final_status,
            final_remaining=final_remaining,
            probs=probs,
            image_url=image_url,
            sim_source_image=sim_source_image,
        )

        _create_alert_if_needed(
            session=session,
            plant_id=plant_id,
            prediction_id=saved_row.id,
            previous_stage=previous_stage,
            current_stage=final_stage,
        )

        final_captured_at = saved_row.captured_at.isoformat()

    except Exception as e:
        session.rollback()
        print("DB insert/update failed:", e)
        final_captured_at = captured_dt.isoformat()

    return SpoilagePredictResponse(
        plant_id=plant_id,
        captured_at=final_captured_at,
        stage=final_stage,
        stage_probs=StageProbs(**probs),
        remaining_days=float(final_remaining),
        status=final_status,
    )


@router.post("/spoilage/stage-only", response_model=StageOnlyResponse)
async def spoilage_stage_only(
    user=Depends(require_user),
    session: Session = Depends(get_session),
    image: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    plant_id: str = Form(...),
    captured_at: str | None = Form(None),
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        plant_id = normalize_plant_id(plant_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    captured_dt = _parse_dt_or_now(captured_at)

    try:
        final_stage, probs = clf.predict(img_bytes, temperature, humidity)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    final_status = make_status(final_stage, probs)

    return StageOnlyResponse(
        plant_id=plant_id,
        captured_at=captured_dt.isoformat(),
        stage=final_stage,
        stage_probs=StageProbs(**probs),
        status=final_status,
    )


@router.post("/spoilage/remaining-days-only", response_model=RemainingDaysOnlyResponse)
def spoilage_remaining_days_only(
    payload: RemainingDaysOnlyRequest,
    user=Depends(require_user),
):
    plant_id = None
    captured_at = payload.captured_at

    if payload.plant_id:
        try:
            plant_id = normalize_plant_id(payload.plant_id)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    if not captured_at or str(captured_at).strip().lower() in ("string", "null", "none"):
        captured_at = datetime.now(timezone.utc).isoformat()

    probs_dict = payload.stage_probs.model_dump()
    remaining = reg.predict(probs_dict, payload.temperature, payload.humidity)

    return RemainingDaysOnlyResponse(
        plant_id=plant_id,
        captured_at=captured_at,
        remaining_days=remaining,
    )


@router.get("/spoilage/predictions", response_model=list[SpoilagePrediction])
def list_predictions(
    session: Session = Depends(get_session),
    limit: int = 20,
):
    stmt = select(SpoilagePrediction).order_by(SpoilagePrediction.captured_at.desc()).limit(limit)
    return session.exec(stmt).all()


@router.get("/spoilage/predictions/by-plant", response_model=list[SpoilagePrediction])
def list_predictions_by_plant(
    plant_id: str = Query(...),
    limit: int = 30,
    session: Session = Depends(get_session),
):
    stmt = (
        select(SpoilagePrediction)
        .where(SpoilagePrediction.plant_id == plant_id)
        .order_by(SpoilagePrediction.captured_at.desc())
        .limit(limit)
    )
    return session.exec(stmt).all()


@router.get("/spoilage/alerts", response_model=list[SpoilageAlert])
def list_spoilage_alerts(
    session: Session = Depends(get_session),
    acknowledged: bool | None = Query(default=None),
    limit: int = 50,
    user=Depends(require_user),
):
    stmt = select(SpoilageAlert).order_by(SpoilageAlert.created_at.desc())

    if acknowledged is not None:
        stmt = stmt.where(SpoilageAlert.is_acknowledged == acknowledged)

    stmt = stmt.limit(limit)
    return session.exec(stmt).all()


@router.post("/spoilage/alerts/{alert_id}/ack", response_model=SpoilageAlert)
def acknowledge_spoilage_alert(
    alert_id: int,
    session: Session = Depends(get_session),
    user=Depends(require_user),
):
    alert = session.get(SpoilageAlert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_acknowledged = True
    alert.acknowledged_at = datetime.now(timezone.utc)

    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


@router.get("/spoilage/recheck")
def list_recheck_recommendations(
    session: Session = Depends(get_session),
    limit: int = 20,
    max_remaining_days: float = 2.0,
    user=Depends(require_user),
):
    latest_rows = _latest_prediction_rows(session, limit=500)

    items = []
    for r in latest_rows:
        remaining = max(0.0, float(r.remaining_days))
        if remaining > max_remaining_days:
            continue
        if r.stage not in ("slightly_aged", "near_spoilage", "spoiled"):
            continue

        urgency = "urgent" if remaining <= 1.0 or r.stage in ("near_spoilage", "spoiled") else "soon"

        items.append(
            {
                "plant_id": r.plant_id,
                "stage": r.stage,
                "remaining_days": remaining,
                "captured_at": r.captured_at.isoformat(),
                "image_url": r.image_url,
                "urgency": urgency,
                "message": f"Plant {r.plant_id} should be rescanned soon. Estimated remaining shelf life is low.",
            }
        )

    items.sort(key=lambda x: (x["remaining_days"], x["captured_at"]))
    return items[:limit]


@router.post("/sim/start")
def sim_start(
    plant_id: str = "P-001",
    interval_sec: int = 15,
    loop: bool = False,
    user=Depends(require_user),
):
    sim.start(
        plant_id=plant_id,
        interval_sec=interval_sec,
        loop=loop,
        reg=reg,
    )
    return {"ok": True, "status": sim.status()}


@router.post("/sim/stop")
def sim_stop(user=Depends(require_user)):
    sim.stop()
    return {"ok": True, "status": sim.status()}


@router.get("/sim/status")
def sim_status(user=Depends(require_user)):
    return sim.status()


@router.get("/sim/sample")
def sim_sample(
    session: Session = Depends(get_session),
    plant_id: str | None = Query(default=None),
    label: str | None = Query(default=None),
    mode: str = Query(default="random"),
    now_iso: str | None = Query(default=None),
    user=Depends(require_user),
):
    pid = None
    if plant_id:
        try:
            pid = normalize_plant_id(plant_id)
        except ValueError:
            pid = plant_id

    if now_iso:
        try:
            now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
        except Exception:
            now = datetime.now(timezone.utc)
    else:
        now = datetime.now(timezone.utc)

    day_id = None

    if mode == "time" and pid:
        first_stmt = (
            select(SpoilagePrediction)
            .where(SpoilagePrediction.plant_id == pid)
            .order_by(SpoilagePrediction.captured_at.asc())
            .limit(1)
        )
        first = session.exec(first_stmt).first()

        if not first:
            day_id = 0
        else:
            base_dt = first.captured_at
            if base_dt.tzinfo is None:
                base_dt = base_dt.replace(tzinfo=timezone.utc)

            day_id = max(
                0,
                (
                    now.astimezone(timezone.utc).date()
                    - base_dt.astimezone(timezone.utc).date()
                ).days,
            )

    row = sim.sample_row(plant_id=pid, label=label, day_id=day_id)
    image_url = f"/sim-images/{row['image_name']}" if row.get("image_name") else None

    return {
        "plant_id": row["plant_id"],
        "temperature": row["temperature"],
        "humidity": row["humidity"],
        "label": row["label"],
        "day_id": row.get("day_id"),
        "capture_date": row.get("capture_date"),
        "image_name": row.get("image_name"),
        "image_url": image_url,
        "remaining_days": row["remaining_days"],
        "mode": mode,
        "now": now.isoformat(),
    }