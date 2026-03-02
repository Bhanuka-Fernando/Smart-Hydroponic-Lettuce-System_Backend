from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from fastapi.responses import Response
from app.services.infer import predict_annotated_image_bytes

from app.db import init_db
from app.schemas import LogCreate
from app.storage import (
    insert_log,
    get_logs_for_plant,
    get_latest_for_plant,
    get_critical_recent,
)
from app.services.infer import predict_from_image_bytes

load_dotenv()

app = FastAPI(title="Disease Service")

COUNTER_FILE = Path("data/plant_counter.txt")
COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)

def _next_plant_id() -> str:
    if not COUNTER_FILE.exists():
        COUNTER_FILE.write_text("0", encoding="utf-8")
    try:
        n = int((COUNTER_FILE.read_text(encoding="utf-8").strip() or "0"))
    except ValueError:
        n = 0
    n += 1
    COUNTER_FILE.write_text(str(n), encoding="utf-8")
    return f"P-{n:04d}"

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True}

# 1) Analyze image (auto plant_id + auto captured_at)
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    result = predict_from_image_bytes(img_bytes)

    # Force server-generated ID + timestamp
    result["plant_id"] = _next_plant_id()
    result["captured_at"] = _now_iso()
    result["image_name"] = image.filename

    return result

# 2) Save to Daily Log (server also ensures id + time exist)
@app.post("/logs")
def save_log(payload: LogCreate):
    if not payload.plant_id:
        payload.plant_id = _next_plant_id()
    if not payload.captured_at:
        payload.captured_at = _now_iso()

    new_id = insert_log(payload)
    return {
        "saved": True,
        "id": new_id,
        "plant_id": payload.plant_id,
        "captured_at": payload.captured_at,
    }

# 3) Dashboard recent critical only (ACT NOW)
@app.get("/dashboard/recent")
def dashboard_recent(limit: int = 5):
    return {"items": get_critical_recent(limit=limit)}

# 4) Plant history (newest first)
@app.get("/plants/{plant_id}/logs")
def plant_logs(plant_id: str, limit: int = 50):
    return {"plant_id": plant_id, "items": get_logs_for_plant(plant_id, limit=limit)}

# 5) Plant latest record
@app.get("/plants/{plant_id}/latest")
def plant_latest(plant_id: str):
    item = get_latest_for_plant(plant_id)
    return {"plant_id": plant_id, "item": item}

@app.post("/predict-annotated")
async def predict_annotated(image: UploadFile = File(...)):
    img_bytes = await image.read()
    out = predict_annotated_image_bytes(img_bytes)
    return Response(content=out, media_type="image/png")