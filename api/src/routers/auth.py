from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..deps import get_db, rate_limit

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── JWT helpers ───────────────────────────────────────────────────────────────


def _create_access_token(user_id: str, email: str, role: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    return jwt.encode(
        {"sub": user_id, "email": email, "role": role, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    _rl: Annotated[None, Depends(rate_limit)],
) -> TokenResponse:
    """Authenticate with email + password. Returns JWT access + opaque refresh token."""
    row = await db.execute(
        sa.text("SELECT id, password_hash, role FROM users WHERE email = :email"),
        {"email": body.email},
    )
    user = row.mappings().first()

    if not user or not _pwd_ctx.verify(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = _create_access_token(str(user["id"]), body.email, user["role"])

    opaque = secrets.token_urlsafe(32)
    token_hash = _hash_token(opaque)
    expires_at = datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days)

    await db.execute(
        sa.text(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) "
            "VALUES (:user_id, :token_hash, :expires_at)"
        ),
        {"user_id": user["id"], "token_hash": token_hash, "expires_at": expires_at},
    )
    await db.commit()

    return TokenResponse(access_token=access_token, refresh_token=opaque)


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AccessTokenResponse:
    """Exchange a valid refresh token for a new access token."""
    token_hash = _hash_token(body.refresh_token)
    row = await db.execute(
        sa.text(
            "SELECT rt.id, u.id AS user_id, u.email, u.role "
            "FROM refresh_tokens rt JOIN users u ON u.id = rt.user_id "
            "WHERE rt.token_hash = :hash AND NOT rt.revoked AND rt.expires_at > NOW()"
        ),
        {"hash": token_hash},
    )
    record = row.mappings().first()

    if not record:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    access_token = _create_access_token(str(record["user_id"]), record["email"], record["role"])
    return AccessTokenResponse(access_token=access_token)


@router.post("/logout", status_code=204)
async def logout(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Revoke the provided refresh token."""
    token_hash = _hash_token(body.refresh_token)
    await db.execute(
        sa.text("UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = :hash"),
        {"hash": token_hash},
    )
    await db.commit()
