from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import AsyncSessionLocal

log = logging.getLogger(__name__)


def check_production_security() -> None:
    """Call from lifespan startup — raises RuntimeError if deployed to production without an API key."""
    if not settings.debug and not settings.api_key_secret:
        raise RuntimeError(
            "API_API_KEY_SECRET must be set in production (debug=False). "
            "All protected endpoints would be publicly accessible without it."
        )


# ── Database ──────────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# ── API key ───────────────────────────────────────────────────────────────────

_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Depends(_api_key_scheme),
) -> str:
    """Validate the X-API-Key header for protected endpoints."""
    if not settings.api_key_secret:
        # No secret configured — skip validation in dev environments only.
        # Production deployments are blocked at startup (see guard above).
        return "dev"
    if api_key != settings.api_key_secret:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ── Rate limiting (fixed-window via atomic Redis Lua) ────────────────────────
# Uses a single atomic Lua round-trip to avoid the INCR+EXPIRE race condition
# where two concurrent requests both see count==1 and both reset the TTL.
# Note: this is a fixed-window counter, not a true sliding window — a client
# can burst 2× limit across a window boundary. Acceptable for this use case;
# upgrade to sorted-set sliding window if stricter enforcement is needed.

_RATE_LIMIT_LUA = """
local c = redis.call('INCR', KEYS[1])
if c == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return c
"""


async def _get_redis_client():  # type: ignore[return]
    """Return an aioredis client if redis is available, else None."""
    client = None
    try:
        import redis.asyncio as aioredis  # type: ignore[import]

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        # Test the connection — from_url() is lazy.
        await client.ping()
        yield client
    except ImportError:
        yield None
    except Exception as exc:
        log.warning("Redis unavailable (%s) — rate limiting disabled", exc)
        yield None
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


async def rate_limit(
    request: Request,
    redis=Depends(_get_redis_client),
) -> None:
    """Fixed-window rate limiter keyed by client IP, atomic via Lua script.

    Falls back gracefully to a no-op when Redis is unavailable.
    """
    if redis is None:
        return

    client_ip: str = request.client.host if request.client else "unknown"
    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_requests
    key = f"rl:{client_ip}:{int(time.time()) // window}"

    count = await redis.eval(_RATE_LIMIT_LUA, 1, key, window)

    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {limit} requests per {window}s",
            headers={"Retry-After": str(window)},
        )
