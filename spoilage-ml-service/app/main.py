# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import router
from app.db import create_db_and_tables

# ✅ ensure table model is registered
import app.models  # noqa

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for mobile testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ absolute base paths (no dependence on terminal working directory)
# app/main.py -> app/ -> spoilage-ml-service/
APP_DIR = Path(__file__).resolve().parent          # .../spoilage-ml-service/app
PROJECT_ROOT = APP_DIR.parent                      # .../spoilage-ml-service

UPLOADS_DIR = PROJECT_ROOT / "uploads"
SIM_IMAGES_DIR = PROJECT_ROOT / "sim_images"

UPLOADS_DIR.mkdir(exist_ok=True)
SIM_IMAGES_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/sim-images", StaticFiles(directory=str(SIM_IMAGES_DIR)), name="sim-images")

app.include_router(router)