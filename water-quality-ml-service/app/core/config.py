from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://hydroponic_user:hydroponic2026@localhost:5432/hydroponic_iot",
    )

    # Service
    port: int = int(os.getenv("PORT", "8006"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    service_name: str = os.getenv("SERVICE_NAME", "water-quality-ml-service")

    # Model artifact paths
    water_model_path: str = os.getenv("WATER_MODEL_PATH", "artifacts/water_status_model.joblib")
    algae_model_path: str = os.getenv("ALGAE_MODEL_PATH", "artifacts/algae_warning_model.joblib")

    # Feature settings
    resample_rule: str = os.getenv("RESAMPLE_RULE", "15min")
    history_minutes: int = int(os.getenv("HISTORY_MINUTES", "90"))


settings = Settings()