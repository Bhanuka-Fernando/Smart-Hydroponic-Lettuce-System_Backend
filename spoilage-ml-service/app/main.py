# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.db import create_db_and_tables
from fastapi.staticfiles import StaticFiles
import os

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

# uploads (existing)
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ✅ NEW: serve simulation images from backend folder
# put images inside: spoilage-ml-service/sim_images/
os.makedirs("sim_images", exist_ok=True)
app.mount("/sim-images", StaticFiles(directory="sim_images"), name="sim-images")

app.include_router(router)