from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://hydroponic_user:password@localhost:5432/hydroponic_iot"

    model_config = SettingsConfigDict(
        env_file=str(SERVICE_DIR / ".env"),
        extra="ignore",
    )

settings = Settings()