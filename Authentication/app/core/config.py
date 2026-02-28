from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # PostgreSQL connection string - override via .env
    DATABASE_URL: str = "postgresql://hydroponic_user:password@localhost:5432/hydroponic_auth"
    SECRET_KEY: str = "key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7 
    GOOGLE_CLIENT_ID: str 
    # tell Pydantic where to read env vars from
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
