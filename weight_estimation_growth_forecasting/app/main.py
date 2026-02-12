from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

from app.core.db import engine
from app.core.db_models import Base

Base.metadata.create_all(bind=engine)


app = FastAPI(title="ML Inference Service", version="0.1.0")

# ✅ allow demo UI to call API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # for demo only (tighten later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"ok": True}
