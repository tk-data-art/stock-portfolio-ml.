"""Configuration module using Pydantic for environment validation."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supabase_url: str = Field(
        default="https://placeholder.supabase.co",
        description="Supabase REST API project URL",
    )
    supabase_key: str = Field(
        default="placeholder-key",
        description="Supabase service role or anon key",
    )

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    default_lookback_years: int = Field(default=5, ge=1, le=20)
    forecast_horizon_days: int = Field(default=365, ge=30, le=730)
    risk_free_rate: float = Field(default=0.045, ge=0.0, le=0.20)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
