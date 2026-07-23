"""
Application configuration using pydantic-settings for environment variable management.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Game Theory Decision Analyzer"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False
    debug: bool = False

    # Provider defaults ("ollama" for local Gemma, or "openai")
    default_provider: str = "ollama"

    # Ollama (local models such as Gemma)
    default_ollama_url: str = "http://localhost:11434"
    default_model_name: str = "gemma4:12b"

    # OpenAI (optional; key read from env so it is never sent from the browser)
    openai_api_key: str = ""
    default_openai_model: str = "gpt-4o"

    # Generation
    temperature: float = 0.2

    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"

    # CORS (comma-separated origins, or * for all)
    cors_origins: str = "*"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()
