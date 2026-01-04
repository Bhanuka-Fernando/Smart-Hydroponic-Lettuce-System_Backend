from fastapi import APIRouter
from app.schemas import WaterReading, WaterStatusResponse, WaterBatchRequest
from app.services.predictor import predict_one, predict_batch

router = APIRouter(prefix="/water", tags=["water"])

@router.get("/health")
def health():
    return {"ok": True, "service": "water-ml-service"}

@router.post("/analyze", response_model=WaterStatusResponse)
def analyze(reading: WaterReading):
    out = predict_one(reading.model_dump())
    out["meta"] = {"note": "single-reading mode (no history)"}
    return out

@router.post("/analyze_batch", response_model=WaterStatusResponse)
def analyze_batch(req: WaterBatchRequest):
    out = predict_batch(req.model_dump())
    return out
