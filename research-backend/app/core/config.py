from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # start with SQLite; later we can override via .env
    DATABASE_URL: str = "sqlite:///./app.db"
    SECRET_KEY: str = "key"
    ALGORITHM: str = "HS256"

    SPOILAGE_MODEL_PATH: str = "app/artifacts/spoilage_stage_classifier.keras"
    SPOILAGE_META_PATH: str = "app/artifacts/spoilage_stage_classifier_meta.json"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7 
    GOOGLE_CLIENT_ID: str 
    # tell Pydantic where to read env vars from
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
