from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.db import engine
from app.core.db_models import Base
from app.api.routes import router as water_router

app = FastAPI(title="Water Quality & Algae Warning Service", version="1.0.0")

# CORS
if settings.cors_origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables
Base.metadata.create_all(bind=engine)

# Routes
app.include_router(water_router)

@app.get("/health")
def root_health():
    return {"status": "ok", "service": settings.service_name}