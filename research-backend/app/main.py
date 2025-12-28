from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.db import init_db
from app.core.config import settings


from app.routers.auth import router as auth_router

from app.routers.spoilage import router as spoilage_router

from app.services.spoilage_classifier import load_spoilage_model, load_meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GOOGLE_CLIENT_ID:", settings.GOOGLE_CLIENT_ID)
    init_db() # runs at startup

     # Load meta + model once
    app.state.spoilage_meta = load_meta(settings.SPOILAGE_META_PATH)
    app.state.spoilage_model = load_spoilage_model(settings.SPOILAGE_MODEL_PATH)

    print("✅ Spoilage meta loaded:", settings.SPOILAGE_META_PATH)
    print("✅ Spoilage model loaded:", settings.SPOILAGE_MODEL_PATH)

    yield

app = FastAPI(
    title="Hydroponic Lettuce Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(spoilage_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}