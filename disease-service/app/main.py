from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from fastapi.responses import Response, FileResponse
import shutil

from app.core.db import engine
from app.core.db_models import Base

from app.schemas import LogCreate
from app.storage import (
    insert_log,
    get_logs_for_plant,
    get_latest_for_plant,
    get_critical_recent,
    get_log_by_id,
    get_all_logs,
)
from app.services.infer import predict_from_image_bytes, predict_annotated_image_bytes

load_dotenv()

app = FastAPI(title="Disease Service")

COUNTER_FILE = Path("data/plant_counter.txt")
COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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


def _safe_filename(name: str) -> str:
    return Path(name).name.replace(" ", "_")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/uploads/{filename}")
def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    return FileResponse(file_path)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    plant_id = _next_plant_id()
    captured_at = _now_iso()

    original_name = image.filename or f"{plant_id}.jpg"
    safe_name = f"{plant_id}_{_safe_filename(original_name)}"
    file_path = UPLOAD_DIR / safe_name

    image.file.seek(0)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    img_bytes = file_path.read_bytes()
    result = predict_from_image_bytes(img_bytes)

    result["plant_id"] = plant_id
    result["captured_at"] = captured_at
    result["image_name"] = original_name
    result["image_path"] = str(file_path).replace("\\", "/")

    return result


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

@app.get("/logs")
def all_logs(limit: int = 50, offset: int = 0):
    return {
        "items": get_all_logs(limit=limit, offset=offset),
        "limit": limit,
        "offset": offset,
    }

@app.get("/dashboard/recent")
def dashboard_recent(limit: int = 5):
    return {"items": get_critical_recent(limit=limit)}


@app.get("/plants/{plant_id}/logs")
def plant_logs(plant_id: str, limit: int = 50):
    return {"plant_id": plant_id, "items": get_logs_for_plant(plant_id, limit=limit)}


@app.get("/plants/{plant_id}/latest")
def plant_latest(plant_id: str):
    item = get_latest_for_plant(plant_id)
    return {"plant_id": plant_id, "item": item}


@app.get("/logs/{log_id}")
def log_by_id(log_id: int):
    item = get_log_by_id(log_id)
    return {"item": item}


@app.post("/predict-annotated")
async def predict_annotated(image: UploadFile = File(...)):
    img_bytes = await image.read()
    out = predict_annotated_image_bytes(img_bytes)
    return Response(content=out, media_type="image/png")