from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from ml.storage.artifact_store import ArtifactStore, R2Config

from .config import settings
from .data_repository import DataRepository
from .logging_cfg import configure_logging
from .routers import auction, leagues, matches, optimizer, quotations, seasons, stats
from .routers import admin_scrape, auth, expert_ratings, mantra, matchday, ml_pipeline, model_metrics
from .routers.intelligence import intelligence_router, predictions_router

configure_logging(settings.log_level)
log = logging.getLogger(__name__)


# ── Application lifespan ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources (Redis client, DataRepository) at startup."""
    redis_client: Any = None
    try:
        import redis.asyncio as aioredis  # type: ignore[import]

        redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
        await redis_client.ping()
        log.info("Redis connection established at %s", settings.redis_url)
    except Exception as exc:
        log.warning("Redis unavailable (%s) — caching disabled", exc)
        redis_client = None

    # Unica porta d'ingresso per lettura/scrittura artefatti (cache-aside
    # locale + R2), condivisa da tutti i router e da DataRepository. Vedi
    # design doc "R2 come source of truth per gli artefatti ML/MANTRA"
    # (2026-08-02), Fase 3 e Fase 4.
    app.state.artifact_store = ArtifactStore(
        local_dir=settings.artifacts_dir,
        r2_config=R2Config(
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket_name=settings.r2_bucket_name,
        ),
    )
    log.info("ArtifactStore initialised (artifacts_dir=%s)", settings.artifacts_dir)

    app.state.repo = DataRepository(
        artifacts_dir=settings.artifacts_dir,
        redis_client=redis_client,
        cache_ttl=settings.cache_ttl_seconds,
        artifact_store=app.state.artifact_store,
    )
    log.info("DataRepository initialised (artifacts_dir=%s)", settings.artifacts_dir)

    yield

    if redis_client is not None:
        await redis_client.aclose()
        log.info("Redis connection closed")


# ── Application factory ───────────────────────────────────────────────────────

app = FastAPI(
    title=settings.title,
    version=settings.version,
    debug=settings.debug,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(matches.router, prefix=settings.api_prefix)
app.include_router(leagues.router, prefix=settings.api_prefix)
app.include_router(seasons.router, prefix=settings.api_prefix)
app.include_router(stats.router, prefix=settings.api_prefix)
app.include_router(predictions_router, prefix=settings.api_prefix)
app.include_router(intelligence_router, prefix=settings.api_prefix)
app.include_router(quotations.quotations_router, prefix=settings.api_prefix)
app.include_router(quotations.id_mapping_router, prefix=settings.api_prefix)
app.include_router(optimizer.router, prefix=settings.api_prefix)
app.include_router(auction.router, prefix=settings.api_prefix)
app.include_router(mantra.router, prefix=settings.api_prefix)
app.include_router(admin_scrape.router, prefix=settings.api_prefix)
app.include_router(matchday.router, prefix=settings.api_prefix)
app.include_router(expert_ratings.router, prefix=settings.api_prefix)
app.include_router(model_metrics.router, prefix=settings.api_prefix)
app.include_router(ml_pipeline.router, prefix=settings.api_prefix)


# ── Health check ───────────────────────────────────────────────────────────────


@app.get(
    f"{settings.api_prefix}/health",
    tags=["health"],
    summary="Health check",
    description="Returns application liveness status and current version.",
    responses={200: {"description": "Service is healthy"}},
)
async def health_check() -> ORJSONResponse:
    return ORJSONResponse({"status": "ok", "version": settings.version})


# ── Global exception handler (RFC 7807 Problem Details) ───────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url)
    return ORJSONResponse(
        status_code=500,
        content={
            "type": "https://tools.ietf.org/html/rfc7807",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please try again later.",
            "instance": str(request.url),
        },
    )
