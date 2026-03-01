from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

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
    image: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    plant_id: str = Form(...),                 # P-001
    captured_at: str | None = Form(None),      # optional
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Empty image")

    # validate / normalize plant id
    try:
        plant_id = normalize_plant_id(plant_id)
    except ValueError as e:
        raise HTTPException(422, str(e))

    # auto timestamp
    if not captured_at:
        captured_at = datetime.now(timezone.utc).isoformat()

    stage, probs = clf.predict(img_bytes, temperature, humidity)
    remaining = reg.predict(probs, temperature, humidity)
    status = make_status(stage, probs)

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
        raise HTTPException(400, "Empty image")

    try:
        plant_id = normalize_plant_id(plant_id)
    except ValueError as e:
        raise HTTPException(422, str(e))

    if not captured_at:
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
            raise HTTPException(422, str(e))

    if not captured_at:
        captured_at = datetime.now(timezone.utc).isoformat()

    # StageProbs is a pydantic model → convert to dict for regressor
    probs_dict = payload.stage_probs.model_dump()
    remaining = reg.predict(probs_dict, payload.temperature, payload.humidity)

    return RemainingDaysOnlyResponse(
        plant_id=plant_id,
        captured_at=captured_at,
        remaining_days=remaining,
    )
