"""
Application configuration using pydantic-settings for environment variable management.
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    app_name: str = "Game Theory Decision Analyzer"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False
    debug: bool = False

    # Ollama Default Settings
    default_ollama_url: str = "http://localhost:11434"
    default_model_name: str = "llama3"

    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"

    # CORS Settings
    cors_origins: str = "*"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
