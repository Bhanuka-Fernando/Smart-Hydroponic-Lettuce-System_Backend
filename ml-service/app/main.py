from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="ML Inference Service", version="0.1.0")

app.include_router(router)

@app.get("/health")
def health():
    return {"ok": True}
