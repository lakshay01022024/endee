"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Centralized application settings loaded from .env file."""

    # ── Endee Vector Database ──────────────────────────
    endee_host: str = "localhost"
    endee_port: int = 8080
    endee_auth_token: str = ""

    # ── Embedding Model ───────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── LLM Configuration ─────────────────────────────
    llm_model: str = "google/flan-t5-base"
    llm_provider: str = "local"           # "local" or "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    # ── Application ────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    index_name: str = "knowledge_base"
    chunk_size: int = 512
    chunk_overlap: int = 50

    @property
    def endee_base_url(self) -> str:
        return f"http://{self.endee_host}:{self.endee_port}/api/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
