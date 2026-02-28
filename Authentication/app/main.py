from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.core.db import init_db
from app.core.config import settings


from app.routers.auth import router as auth_router
from app.routers.users import router as users_router

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

# Mount static files for avatars
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth_router)
app.include_router(users_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}