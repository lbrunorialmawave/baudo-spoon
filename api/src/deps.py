from __future__ import annotations

import time
from collections.abc import AsyncGenerator

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import AsyncSessionLocal

# ── Database ──────────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# ── API key ───────────────────────────────────────────────────────────────────

_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Depends(_api_key_scheme),
) -> str:
    """Validate the X-API-Key header for protected /intelligence endpoints."""
    if not settings.api_key_secret:
        # No secret configured — skip validation in dev environments.
        return "dev"
    if api_key != settings.api_key_secret:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ── Rate limiting (Redis sliding-window INCR/EXPIRE) ─────────────────────────


async def _get_redis_client():  # type: ignore[return]
    """Return an aioredis client if redis is available, else None."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import]

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        yield client
        await client.aclose()
    except ImportError:
        yield None


async def rate_limit(
    request: Request,
    redis=Depends(_get_redis_client),
) -> None:
    """Sliding-window rate limiter keyed by client IP.

    Falls back gracefully to a no-op when Redis is unavailable.
    """
    if redis is None:
        return

    client_ip: str = request.client.host if request.client else "unknown"
    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_requests
    key = f"rl:{client_ip}:{int(time.time()) // window}"

    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window)

    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {limit} requests per {window}s",
            headers={"Retry-After": str(window)},
        )
