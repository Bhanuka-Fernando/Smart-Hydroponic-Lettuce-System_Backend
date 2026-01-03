from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="water-ml-service")
app.include_router(router)
