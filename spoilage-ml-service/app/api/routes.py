from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.core.security import require_user
from app.core.config import settings
from app.schemas import SpoilagePredictResponse
from app.services.spoilage_classifier import SpoilageClassifier
from app.services.remaining_days import RemainingDaysRegressor
from app.services.postprocess import make_status
from app.schemas import StageOnlyResponse, RemainingDaysOnlyRequest, RemainingDaysOnlyResponse, StageProbs


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
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Empty image")

    stage, probs = clf.predict(img_bytes, temperature, humidity)
    remaining = reg.predict(probs, temperature, humidity)
    status = make_status(stage, probs)

    return SpoilagePredictResponse(
        stage=stage,
        stage_probs=probs,
        remaining_days=remaining,
        status=status,
    )

@router.post("/spoilage/stage-only", response_model=StageOnlyResponse)
async def spoilage_stage_only(
    user=Depends(require_user),
    image: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Empty image")

    stage, probs = clf.predict(img_bytes, temperature, humidity)
    status = make_status(stage, probs)

    return StageOnlyResponse(stage=stage, stage_probs=StageProbs(**probs), status=status)


@router.post("/spoilage/remaining-days-only", response_model=RemainingDaysOnlyResponse)
def spoilage_remaining_days_only(
    payload: RemainingDaysOnlyRequest,
    user=Depends(require_user),
):
    remaining = reg.predict(payload.stage_probs, payload.temperature, payload.humidity)
    return RemainingDaysOnlyResponse(remaining_days=remaining)

