from datetime import datetime, timezone
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import SpoilagePrediction
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
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    # ✅ validate / normalize plant id FIRST
    try:
        plant_id = normalize_plant_id(plant_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ✅ auto timestamp (Swagger often sends "string")
    if not captured_at or captured_at.strip().lower() in ("string", "null", "none"):
        captured_at = datetime.now(timezone.utc).isoformat()

    # ✅ save uploaded image to /uploads and generate URL
    Path("uploads").mkdir(exist_ok=True)
    filename = f"{plant_id}_{uuid.uuid4().hex}.jpg"
    file_path = Path("uploads") / filename
    file_path.write_bytes(img_bytes)
    image_url = f"/uploads/{filename}"

    # ✅ run models
    stage, probs = clf.predict(img_bytes, temperature, humidity)
    remaining = reg.predict(probs, temperature, humidity)
    status = make_status(stage, probs)

    # ✅ SAVE TO DB (do not break response if insert fails)
    try:
        try:
            dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)

        row = SpoilagePrediction(
            plant_id=plant_id,
            captured_at=dt,
            temperature=float(temperature),
            humidity=float(humidity),
            stage=stage,
            status=status,
            remaining_days=float(remaining),
            p_fresh=float(probs["fresh"]),
            p_slightly_aged=float(probs["slightly_aged"]),
            p_near_spoilage=float(probs["near_spoilage"]),
            p_spoiled=float(probs["spoiled"]),
            image_url=image_url,  # ✅ IMPORTANT
        )
        session.add(row)
        session.commit()
        session.refresh(row)
    except Exception as e:
        session.rollback()
        print("DB insert failed:", e)

    return SpoilagePredictResponse(
        plant_id=plant_id,
        captured_at=captured_at,
        stage=stage,
        stage_probs=StageProbs(**probs),
        remaining_days=remaining,
        status=status,
    )


@router.post("/spoilage/stage-only", response_model=StageOnlyResponse)
async def spoilage_stage_only(
    user=Depends(require_user),
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

    if not captured_at or captured_at.strip().lower() in ("string", "null", "none"):
        captured_at = datetime.now(timezone.utc).isoformat()

    stage, probs = clf.predict(img_bytes, temperature, humidity)
    status = make_status(stage, probs)

    return StageOnlyResponse(
        plant_id=plant_id,
        captured_at=captured_at,
        stage=stage,
        stage_probs=StageProbs(**probs),
        status=status,
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
    stmt = select(SpoilagePrediction).order_by(SpoilagePrediction.id.desc()).limit(limit)
    return session.exec(stmt).all()