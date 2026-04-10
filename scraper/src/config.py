from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScraperSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCRAPER_",
        env_file=".env",
        extra="ignore",
    )

    # Optional locally — required only when --no-db is NOT passed.
    # Set via SCRAPER_DATABASE_URL environment variable.
    database_url: Optional[str] = Field(None, description="PostgreSQL connection URL")
    output_dir: Path = Path("downloaded_files/output")
    log_level: str = "INFO"


settings = ScraperSettings()
