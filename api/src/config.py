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
    ml_coverage_warning_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0,
        description="Warning threshold for active-list ML coverage (env: API_ML_COVERAGE_WARNING_THRESHOLD)",
    )

    artifacts_dir: Path = Field(
        default=Path("ml/artifacts"),
        description="Directory containing ML pipeline output JSON artifacts",
    )
    # Mirrors MLConfig.reliability_weight_mode (plan-limited-cohort-patches G3).
    reliability_weight_mode: str = Field(
        default="continuous",
        description="Decision reliability weight mode: continuous | bucket",
    )

    # ── Artifact storage (Cloudflare R2) ────────────────────────────────────
    r2_endpoint_url: str | None = Field(default=None, description="https://<account_id>.r2.cloudflarestorage.com")
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket_name: str = Field(default="baudo-spoon-ml-artifacts")

    # Redis — optional; caching is disabled when not provided.
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for intelligence endpoint cache",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="TTL for Redis-cached ML results (seconds)",
    )

    # GitHub Actions trigger for ML training (see .github/workflows/ml-training.yml).
    # The ml pipeline needs dependencies (xgboost, shap, matplotlib, ...) the API's
    # own image deliberately doesn't install, so training can't run as a local
    # subprocess in production — this dispatches the existing, already-secrets
    # -configured GitHub Actions workflow instead. Set API_GITHUB_TOKEN to a PAT
    # with `actions:write` on the repo.
    github_token: str | None = Field(default=None, description="GitHub PAT with actions:write, for triggering ML training")
    github_repo: str = Field(default="lbrunorialmawave/baudo-spoon", description="owner/repo for the ML training workflow")
    github_default_branch: str = Field(default="main", description="Branch to dispatch the ML training workflow on")

    # Security — /v1/intelligence endpoints require this key via X-API-Key header.
    api_key_secret: str = Field(
        default="",
        description="Secret token for /intelligence route authentication (set API_API_KEY_SECRET)",
    )

    # JWT auth (set API_JWT_SECRET to a long random string in production)
    jwt_secret: str = Field(
        default="dev-insecure-secret-change-me",
        description="HMAC-SHA256 secret for signing JWTs (set API_JWT_SECRET)",
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 30

    # Rate limiting (sliding-window via Redis INCR + EXPIRE)
    rate_limit_requests: int = Field(
        default=60,
        description="Maximum requests per window per IP",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate-limit sliding-window size in seconds",
    )

    # Monte Carlo: hard ceiling for n_simulations on every path (sync + async jobs).
    optimizer_max_simulations: int = Field(
        default=1000,
        ge=1,
        description=(
            "Hard cap for monte_carlo.n_simulations on sync and async paths "
            "(env: API_OPTIMIZER_MAX_SIMULATIONS)."
        ),
    )
    optimizer_mc_default_enabled: bool = Field(default=False)
    optimizer_saa_timeout_seconds: float = Field(default=120.0)
    # Above this N, sync endpoints (/optimize/multi, /optimize/single) reject
    # saa_frequency requests and instruct the client to use POST /optimize/jobs.
    # mean_std is exempt (O(1) ILP solves). Prefer threshold <= max_simulations.
    optimizer_async_threshold: int = Field(
        default=50,
        ge=1,
        description=(
            "Max n_simulations allowed on the synchronous path for saa_frequency "
            "(env: API_OPTIMIZER_ASYNC_THRESHOLD). Higher N must use POST /optimize/jobs."
        ),
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
