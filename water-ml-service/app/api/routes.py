from fastapi import APIRouter
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.whs import compute_whs
from app.services.anomaly import infer_anomaly
from app.services.recommend import recommend

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "water-ml-service"}

@router.post("/water/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    latest = req.readings[-1]

    whs, level = compute_whs(latest.pH, latest.EC_mS_cm, latest.temp_C, latest.do_mg_L)
    anom_score, th, flag = infer_anomaly(req.tank_id, [r.model_dump() for r in req.readings])

    recs = recommend(
        pH=latest.pH,
        EC=latest.EC_mS_cm,
        temp=latest.temp_C,
        DO=latest.do_mg_L,
        risk_level=level,
        anomaly_flag=flag,
    )

    return AnalyzeResponse(
        tank_id=req.tank_id,
        timestamp=latest.timestamp,
        WHS_0_100=whs,
        risk_level=level,
        anom_score_0_1=anom_score,
        threshold=th,
        anomaly_flag=flag,
        persistence="2of3",
        recommendations=recs,
    )
