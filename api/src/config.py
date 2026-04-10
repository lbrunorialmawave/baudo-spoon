from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
    )

    # Required — set via API_DATABASE_URL environment variable.
    database_url: str = Field(..., description="PostgreSQL connection URL (sync DSN; asyncpg variant derived automatically)")
    api_prefix: str = "/api/v1"
    debug: bool = False
    log_level: str = "INFO"
    title: str = "FBref Data Platform API"
    version: str = "1.0.0"

    # ML artifacts
    artifacts_dir: Path = Field(
        default=Path("ml/artifacts"),
        description="Directory containing ML pipeline output JSON artifacts",
    )

    # Redis — optional; caching is disabled when not provided.
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for intelligence endpoint cache",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="TTL for Redis-cached ML results (seconds)",
    )

    # Security — /v1/intelligence endpoints require this key via X-API-Key header.
    api_key_secret: str = Field(
        default="",
        description="Secret token for /intelligence route authentication (set API_API_KEY_SECRET)",
    )

    # Rate limiting (sliding-window via Redis INCR + EXPIRE)
    rate_limit_requests: int = Field(
        default=60,
        description="Maximum requests per window per IP",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate-limit sliding-window size in seconds",
    )

    @computed_field  # type: ignore[misc]
    @property
    def async_database_url(self) -> str:
        """asyncpg-compatible DSN derived from the sync database_url."""
        url = self.database_url
        for sync_prefix in (
            "postgresql+psycopg2://",
            "postgresql+psycopg://",
            "postgresql://",
            "postgres://",
        ):
            if url.startswith(sync_prefix):
                return "postgresql+asyncpg://" + url[len(sync_prefix):]
        return url


settings = APISettings()
