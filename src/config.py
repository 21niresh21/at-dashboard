"""Application configuration module.

Loads settings from environment variables with sensible defaults.
Uses pydantic-settings for validation and type safety.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file (src/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Environment(str, Enum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Central application settings.

    Values are read from environment variables (case-insensitive)
    and fall back to the defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────────────────
    app_name: str = "AT Dashboard"
    app_env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    # ── Paths ────────────────────────────────────────────────────────────────
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    logs_dir: Path = _PROJECT_ROOT / "logs"

    # ── Database ─────────────────────────────────────────────────────────────
    db_path: Path = _PROJECT_ROOT / "data" / "at_instruments.db"

    # ── SmartAPI (AngelOne) ──────────────────────────────────────────────────
    smartapi_master_url: str = (
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    )
    smartapi_api_key: str = Field(
        default="", validation_alias="ANGEL_ONE_API_KEY"
    )
    smartapi_client_id: str = Field(
        default="", validation_alias="ANGEL_ONE_CLIENT_CODE"
    )
    smartapi_password: str = Field(
        default="", validation_alias="ANGEL_ONE_PIN"
    )
    smartapi_totp_secret: str = Field(
        default="", validation_alias="ANGEL_ONE_TOTP_SECRET"
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT


# Singleton — import and use directly
settings = Settings()
