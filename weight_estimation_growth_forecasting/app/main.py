from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as infer_router
from app.api.plants import router as plants_router

from app.core.db import engine
from app.core.db_models import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ML Inference Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ include ONCE only
app.include_router(infer_router)
app.include_router(plants_router)

@app.get("/health")
def health():
    return {"ok": True}