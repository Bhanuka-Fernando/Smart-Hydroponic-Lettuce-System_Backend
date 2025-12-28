from fastapi import APIRouter, File, Form, UploadFile, Request, Depends
from app.schemas.spoilage import SpoilageClassifyResponse
from app.services.spoilage_classifier import predict_spoilage  # (your service import)
from app.routers.auth import get_current_user  # ✅ correct import

router = APIRouter(prefix="/spoilage", tags=["spoilage"])

@router.post("/classify", response_model=SpoilageClassifyResponse)
async def classify_spoilage(
    request: Request,
    image: UploadFile = File(...),
    temp_c: float = Form(...),
    humidity_pct: float = Form(...),
    _user = Depends(get_current_user),  # ✅ uses your existing auth dependency
):
    image_bytes = await image.read()

    model = request.app.state.spoilage_model
    meta = request.app.state.spoilage_meta

    stage, conf, probs = predict_spoilage(model, meta, image_bytes, temp_c, humidity_pct)
    return SpoilageClassifyResponse(stage=stage, confidence=conf, probs=probs)
