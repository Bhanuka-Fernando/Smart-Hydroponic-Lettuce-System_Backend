from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db() # runs at startup
    yield

app = FastAPI(
    title="Hydroponic Lettuce Backend",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    return {"status": "ok"}