from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # start with SQLite; later we can override via .env
    DATABASE_URL: str = "sqlite:///./app.db"

    # tell Pydantic where to read env vars from
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
