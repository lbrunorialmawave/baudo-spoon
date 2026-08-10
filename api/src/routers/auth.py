from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
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


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


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


async def _store_refresh_token(
    db: AsyncSession, user_id: str, family_id: uuid.UUID
) -> str:
    """Generate, hash and persist a new opaque refresh token within ``family_id``.

    Does NOT commit — caller controls the transaction boundary so that
    rotation (revoke old + insert new) is atomic.
    """
    opaque = secrets.token_urlsafe(32)
    token_hash = _hash_token(opaque)
    expires_at = datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days)

    await db.execute(
        sa.text(
            "INSERT INTO refresh_tokens (user_id, token_hash, family_id, expires_at) "
            "VALUES (:user_id, :token_hash, :family_id, :expires_at)"
        ),
        {
            "user_id": user_id,
            "token_hash": token_hash,
            "family_id": family_id,
            "expires_at": expires_at,
        },
    )
    return opaque


async def _issue_tokens(
    db: AsyncSession, user_id: str, email: str, role: str
) -> TokenResponse:
    """Create an access token and start a brand-new refresh token family.

    Shared by ``/login`` and ``/register`` so both flows issue tokens
    identically. A fresh login always starts a new rotation family — it
    must never be able to invalidate an existing session on another device.
    """
    access_token = _create_access_token(user_id, email, role)
    refresh_token = await _store_refresh_token(db, user_id, family_id=uuid.uuid4())
    await db.commit()

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


async def _revoke_family(db: AsyncSession, family_id: uuid.UUID, reason: str) -> None:
    await db.execute(
        sa.text(
            "UPDATE refresh_tokens "
            "SET revoked = TRUE, revoked_at = NOW(), revoked_reason = :reason "
            "WHERE family_id = :family_id AND NOT revoked"
        ),
        {"family_id": family_id, "reason": reason},
    )


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

    return await _issue_tokens(db, str(user["id"]), body.email, user["role"])


@router.post("/register", response_model=TokenResponse)
async def register(
    body: RegisterRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    _rl: Annotated[None, Depends(rate_limit)],
) -> TokenResponse:
    """Public self-service registration.

    Always creates a ``role='member'`` account — the client cannot request a
    role. Admin accounts are provisioned out-of-band. Auto-logs-in on
    success (same response shape as ``/login``).
    """
    password_hash = _pwd_ctx.hash(body.password)

    try:
        row = await db.execute(
            sa.text(
                "INSERT INTO users (email, password_hash, role) "
                "VALUES (:email, :password_hash, 'member') "
                "RETURNING id"
            ),
            {"email": body.email, "password_hash": password_hash},
        )
        user_id = row.scalar_one()
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")

    return await _issue_tokens(db, str(user_id), body.email, "member")


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    _rl: Annotated[None, Depends(rate_limit)],
) -> TokenResponse:
    """Exchange a refresh token for a new access + refresh token pair (rotation).

    Security model:
    - Each refresh token is single-use. On success, it is revoked and a new
      one is issued in the same ``family_id``.
    - If a token that is already revoked is presented again, that is proof
      of reuse (stolen or previously-consumed token) — the entire family is
      revoked immediately, forcing re-login on every device sharing it.
    - ``SELECT ... FOR UPDATE`` serializes concurrent refreshes of the same
      token so two racing requests can't both "win" a rotation and leave an
      unrevoked duplicate in circulation.
    """
    token_hash = _hash_token(body.refresh_token)

    row = await db.execute(
        sa.text(
            "SELECT rt.id, rt.user_id, rt.family_id, rt.revoked, rt.expires_at, "
            "       u.email, u.role "
            "FROM refresh_tokens rt "
            "JOIN users u ON u.id = rt.user_id "
            "WHERE rt.token_hash = :hash "
            "FOR UPDATE OF rt"
        ),
        {"hash": token_hash},
    )
    record = row.mappings().first()

    if not record:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if record["revoked"]:
        # Reuse of an already-consumed (or already-revoked) token: treat the
        # whole family as compromised rather than trusting this one token.
        log.warning(
            "Refresh token reuse detected - revoking family %s (user %s)",
            record["family_id"],
            record["user_id"],
        )
        await _revoke_family(db, record["family_id"], reason="reuse_detected")
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if record["expires_at"] <= datetime.now(UTC):
        raise HTTPException(status_code=401, detail="Refresh token expired")

    # Rotate: revoke the presented token, issue a new one in the same family.
    await db.execute(
        sa.text(
            "UPDATE refresh_tokens "
            "SET revoked = TRUE, revoked_at = NOW(), revoked_reason = 'rotated' "
            "WHERE id = :id"
        ),
        {"id": record["id"]},
    )
    new_refresh_token = await _store_refresh_token(
        db, str(record["user_id"]), family_id=record["family_id"]
    )
    access_token = _create_access_token(str(record["user_id"]), record["email"], record["role"])
    await db.commit()

    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout", status_code=204)
async def logout(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Revoke the single session tied to the provided refresh token."""
    token_hash = _hash_token(body.refresh_token)
    await db.execute(
        sa.text(
            "UPDATE refresh_tokens "
            "SET revoked = TRUE, revoked_at = NOW(), revoked_reason = 'logout' "
            "WHERE token_hash = :hash AND NOT revoked"
        ),
        {"hash": token_hash},
    )
    await db.commit()


@router.post("/logout-all", status_code=204)
async def logout_all(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Revoke every active session (every family) belonging to this user.

    Useful after a suspected compromise, or as a "log out of all devices"
    self-service action. Identity is derived from the refresh token itself
    (not from an access token) so it works even once the access token has
    already expired.
    """
    token_hash = _hash_token(body.refresh_token)
    row = await db.execute(
        sa.text("SELECT user_id FROM refresh_tokens WHERE token_hash = :hash"),
        {"hash": token_hash},
    )
    record = row.mappings().first()
    if not record:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    await db.execute(
        sa.text(
            "UPDATE refresh_tokens "
            "SET revoked = TRUE, revoked_at = NOW(), revoked_reason = 'logout_all' "
            "WHERE user_id = :user_id AND NOT revoked"
        ),
        {"user_id": record["user_id"]},
    )
    await db.commit()
