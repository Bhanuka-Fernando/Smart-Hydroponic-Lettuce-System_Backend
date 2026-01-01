from fastapi import APIRouter, UploadFile, File, Form

from app.schemas import (
    InferRequest,
    InferResponse,
    ForecastRequest,
    ForecastResponse,
)
from app.services.vision import get_proj_area_and_diam
from app.services.leaf_area import leaf_area_from_proj
from app.services.weight import predict_weight_g
from app.services.growth import predict_tomorrow, forecast_n_days

router = APIRouter(prefix="/infer", tags=["inference"])


@router.post("/today", response_model=InferResponse)
async def infer_today(
    payload_json: str = Form(...),
    image: UploadFile = File(...),
    depth: UploadFile = File(...),
):
    payload = InferRequest.model_validate_json(payload_json)

    rgb_bytes = await image.read()
    depth_bytes = await depth.read()

    A_proj_cm2, D_proj_cm, Z_m = get_proj_area_and_diam(rgb_bytes, depth_bytes)

    A_des_cm2 = leaf_area_from_proj(A_proj_cm2, D_proj_cm)
    W_today_g = predict_weight_g(A_des_cm2, D_proj_cm)

    # must come from DB for good forecasts; fallback = no growth info
    A_prev = payload.A_prev_cm2 if payload.A_prev_cm2 is not None else A_proj_cm2

    # ✅ correct call (deltaA/RGR computed inside growth.py)
    A_tmr, D_tmr = predict_tomorrow(
        dap=payload.dap,
        A_t_cm2=A_proj_cm2,
        D_t_cm=D_proj_cm,
        A_prev_cm2=A_prev,
        sensors=payload.sensors,
    )

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
    )


@router.post("/forecast", response_model=ForecastResponse)
async def infer_forecast(payload: ForecastRequest):
    points = forecast_n_days(
        dap_start=payload.dap,
        A_prev_cm2=payload.A_prev_cm2,
        A_t_cm2=payload.A_t_cm2,
        D_t_cm=payload.D_t_cm,
        sensors=payload.sensors,
        n_days=payload.n_days,
    )
    return ForecastResponse(points=points)
