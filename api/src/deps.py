from __future__ import annotations

import hmac
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import AsyncSessionLocal

log = logging.getLogger(__name__)


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


# ── JWT role-based auth ───────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)

_ROLE_HIERARCHY = {"member": 0, "admin": 1}


def require_role(required: str):
    """FastAPI dependency that requires a JWT Bearer token with at least the given role.

    Role hierarchy: admin > member.
    Raises 401 if token missing/invalid, 403 if role insufficient.
    """
    async def _dep(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
        ],
    ) -> dict:
        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
            )
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        role = payload.get("role", "")
        if _ROLE_HIERARCHY.get(role, -1) < _ROLE_HIERARCHY.get(required, 0):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return payload

    return _dep


# ── HMAC-signed service requests (M2M, replay-resistant) ─────────────────────
# Alternative to the static X-API-Key for callers that sign each request
# instead of putting the raw secret on the wire (e.g. scripts/season_refresh.py).
# Signature = HMAC-SHA256(shared_secret, f"{timestamp}:{METHOD}:{path}"),
# sent as X-API-Timestamp / X-API-Signature. Bound to method+path+time so a
# captured request can't be replayed elsewhere or later.
_SIGNATURE_WINDOW_SECONDS = 300


def _verify_signed_request(request: Request) -> bool:
    timestamp = request.headers.get("X-API-Timestamp")
    signature = request.headers.get("X-API-Signature")
    if not timestamp or not signature or not settings.api_key_secret:
        return False
    try:
        skew = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    if skew > _SIGNATURE_WINDOW_SECONDS:
        return False
    message = f"{timestamp}:{request.method.upper()}:{request.url.path}".encode()
    expected = hmac.new(settings.api_key_secret.encode(), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


async def require_admin(
    request: Request,
    api_key: Annotated[str | None, Depends(_api_key_scheme)] = None,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> dict:
    """Accept a valid API key, an HMAC-signed request, or an admin JWT.

    Used on scraper-trigger endpoints so scheduled jobs can authenticate with
    the static API key (or a signed request) while human admins use their JWT.
    """
    # API key path (legacy cron / machine-to-machine — raw secret in header)
    if api_key and settings.api_key_secret and api_key == settings.api_key_secret:
        return {"role": "admin", "sub": "api-key"}

    # Signed-request path (preferred M2M — no raw secret on the wire, replay-resistant)
    if _verify_signed_request(request):
        return {"role": "admin", "sub": request.headers.get("X-Service-Id", "signed-service")}

    # JWT path
    if credentials:
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
            )
            if _ROLE_HIERARCHY.get(payload.get("role", ""), -1) >= _ROLE_HIERARCHY["admin"]:
                return payload
            raise HTTPException(status_code=403, detail="Admin role required")
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    raise HTTPException(
        status_code=401,
        detail="Authentication required (API key or Bearer token)",
        headers={"WWW-Authenticate": "Bearer"},
    )


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
