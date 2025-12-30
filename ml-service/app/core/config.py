from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str = "HS256"

    FX: float
    FY: float
    DEPTH_SCALE: float
    SEG_THRESHOLD: float = 0.5
    USE_IMAGENET_NORM: bool = False 


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
