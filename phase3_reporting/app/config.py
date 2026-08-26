"""
config.py
---------
Centralised application configuration.

All tunable values are read from environment variables (or a .env file
loaded automatically by Pydantic-Settings).  Every consumer imports this
singleton instead of reaching for os.environ directly.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Priority order (highest → lowest):
        1.  Actual environment variables
        2.  Values in the .env file
        3.  Default values defined below
    """

    # Logging settings
    log_file: str = "logs/phase3.log"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Ollama / LLM
    # ------------------------------------------------------------------ #
    # These defaults make a fresh local installation work, while every
    # deployment can override them through OLLAMA_HOST / OLLAMA_MODEL.  Do
    # not use file-system paths here: Ollama manages models per user.
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_temperature: float = 0.2   # kept low for determinism
    ollama_timeout: int = 120         # seconds
    ollama_start_timeout: int = 30    # seconds to wait for a local server
    ollama_auto_pull: bool = True     # used by the optional setup helper
    ollama_command: str = "ollama"    # CLI name/path; override if not on PATH

    # ------------------------------------------------------------------ #
    # FastAPI
    # ------------------------------------------------------------------ #
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    fastapi_debug: bool = False

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    reports_dir: str = "reports"

    # ------------------------------------------------------------------ #
    # Application behaviour
    # ------------------------------------------------------------------ #
    # When True the LLM is completely skipped and the template text is
    # returned directly.  Useful for offline / CI testing.
    skip_llm: bool = False

    # Optional Radiologist / Institution metadata (used in future prints)
    institution_name: Optional[str] = "PRISM Radiology Centre"
    radiologist_name: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
