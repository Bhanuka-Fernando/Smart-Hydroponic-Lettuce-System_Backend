from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.db import init_db
from app.core.config import settings


from app.routers.auth import router as auth_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GOOGLE_CLIENT_ID:", settings.GOOGLE_CLIENT_ID)
    init_db() # runs at startup
    yield

app = FastAPI(
    title="Hydroponic Lettuce Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}