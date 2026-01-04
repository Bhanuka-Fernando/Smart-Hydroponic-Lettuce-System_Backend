from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Water ML Service")
app.include_router(router)
