"""Tests for application configuration."""

from src.config import Environment, Settings


def test_default_settings() -> None:
    """Settings should load with sensible defaults when no .env is present."""
    s = Settings()
    assert s.app_name == "AT Dashboard"
    assert s.app_env == Environment.DEVELOPMENT
    assert s.log_level == "INFO"


def test_is_development_property() -> None:
    s = Settings()
    assert s.is_development is True
    assert s.is_production is False
