from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    database_url: str = "sqlite:///./chat.db"
    temperature: float = 0.3
    max_history_messages: int = 20


settings = Settings()
