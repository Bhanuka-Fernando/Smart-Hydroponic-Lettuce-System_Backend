from fastapi import APIRouter, UploadFile, File, Depends, Form
from app.schemas import InferRequest, InferResponse
from app.core.security import require_user
from app.services.vision import get_proj_area_and_diam
from app.services.leaf_area import leaf_area_from_proj
from app.services.weight import predict_weight_g
from app.services.growth import predict_tomorrow

router = APIRouter(prefix="/infer", tags=["inference"])

@router.post("/today", response_model=InferResponse)
async def infer_today(payload_json: str = Form(...), image: UploadFile = File(...), user=Depends(require_user)):
    payload = InferRequest.model_validate_json(payload_json)
    img_bytes = await image.read()

    A_proj_cm2, D_proj_cm = get_proj_area_and_diam(img_bytes)
    A_leaf_cm2 = leaf_area_from_proj(A_proj_cm2, D_proj_cm)
    W_today_g = predict_weight_g(A_leaf_cm2, D_proj_cm)

    A_tmr, D_tmr = predict_tomorrow(
        dap=payload.dap,
        A_t_cm2=A_proj_cm2,
        D_t_cm=D_proj_cm,
        deltaA_cm2=payload.deltaA_cm2,
        RGR=payload.RGR,
        sensors=payload.sensors,
    )

    A_leaf_tmr = leaf_area_from_proj(A_tmr, D_tmr)
    W_tmr_g = predict_weight_g(A_leaf_tmr, D_tmr)

    return InferResponse(
        A_proj_cm2=A_proj_cm2,
        D_proj_cm=D_proj_cm,
        A_leaf_cm2=A_leaf_cm2,
        W_today_g=W_today_g,
        A_proj_tmr_cm2=A_tmr,
        D_proj_tmr_cm=D_tmr,
        W_tmr_g=W_tmr_g,
    )
