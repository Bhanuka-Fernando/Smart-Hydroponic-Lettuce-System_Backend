from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "water-quality-ml-service")
    port: int = int(os.getenv("PORT", "8006"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")

    water_model_path: str = os.getenv("WATER_MODEL_PATH", "artifacts/water_status_model.joblib")
    algae_model_path: str = os.getenv("ALGAE_MODEL_PATH", "artifacts/algae_warning_model.joblib")

    resample_rule: str = os.getenv("RESAMPLE_RULE", "15min")

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    history_minutes: int = int(os.getenv("HISTORY_MINUTES", "90"))

settings = Settings()