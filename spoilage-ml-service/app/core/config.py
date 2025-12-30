from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "spoilage-ml-service"

    JWT_SECRET: str = "change_me"
    JWT_ALGORITHM: str = "HS256"

    STAGE_MODEL_PATH: str = "artifacts/spoilage_stage_classifier.keras"
    STAGE_META_PATH: str = "artifacts/spoilage_stage_classifier_meta.json"

    REG_MODEL_PATH: str = "artifacts/remaining_days_linear.joblib"
    REG_META_PATH: str = "artifacts/remaining_days_linear_meta.json"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
