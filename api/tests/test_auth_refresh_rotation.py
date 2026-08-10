"""Test per la rotation dei refresh token e la reuse detection.

Monta solo ``auth.router`` con ``get_db`` sovrascritto da una sessione fake
in-memory (nessun Postgres reale richiesto), seguendo la stessa convenzione
di isolamento di ``test_auction_router.py``. Ogni test fornisce uno
"script" di risultati che simula, in ordine, le righe restituite dalle
``execute()`` che il codice di produzione esegue — così il test verifica il
comportamento dell'endpoint, non i dettagli SQL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import auth as auth_router


# ---------------------------------------------------------------------------
# Fake AsyncSession
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, row: dict[str, Any] | None):
        self._row = row

    def mappings(self) -> "_FakeResult":
        return self

    def first(self) -> dict[str, Any] | None:
        return self._row

    def scalar_one(self) -> Any:
        return self._row


class FakeSession:
    """Restituisce i risultati di ``script`` in ordine, uno per ogni ``execute()``."""

    def __init__(self, script: list[dict[str, Any] | None]):
        self._script = list(script)
        self.executed_sql: list[str] = []
        self.committed = False
        self.rolled_back = False

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        self.executed_sql.append(str(stmt))
        row = self._script.pop(0) if self._script else None
        return _FakeResult(row)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


def _make_client(session: FakeSession) -> TestClient:
    app = FastAPI()
    app.include_router(auth_router.router)

    async def _override_get_db():
        yield session

    async def _override_rate_limit() -> None:
        return None

    app.dependency_overrides[auth_router.get_db] = _override_get_db
    app.dependency_overrides[auth_router.rate_limit] = _override_rate_limit
    return TestClient(app, raise_server_exceptions=False)


def _active_token_row(*, family_id: uuid.UUID | None = None, expires_in_days: int = 30) -> dict[str, Any]:
    return {
        "id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "family_id": family_id or uuid.uuid4(),
        "revoked": False,
        "expires_at": datetime.now(UTC) + timedelta(days=expires_in_days),
        "email": "user@example.com",
        "role": "member",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_refresh_rotates_token_and_returns_new_pair() -> None:
    """Una refresh valida ruota il token: la risposta contiene un refresh_token
    diverso da quello presentato, e la family viene preservata (implicito nel
    fatto che l'INSERT successivo riusa record["family_id"])."""
    row = _active_token_row()
    session = FakeSession(script=[row])  # 1 SELECT ... FOR UPDATE
    client = _make_client(session)

    resp = client.post("/auth/refresh", json={"refresh_token": "old-token-value"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["refresh_token"] != "old-token-value"
    assert session.committed is True
    # SELECT ... FOR UPDATE, poi UPDATE (rotate) e INSERT (nuovo token).
    assert len(session.executed_sql) == 3
    assert "FOR UPDATE" in session.executed_sql[0]
    assert "UPDATE refresh_tokens" in session.executed_sql[1]
    assert "INSERT INTO refresh_tokens" in session.executed_sql[2]


def test_refresh_with_unknown_token_returns_401() -> None:
    session = FakeSession(script=[None])
    client = _make_client(session)

    resp = client.post("/auth/refresh", json={"refresh_token": "nonexistent"})

    assert resp.status_code == 401


def test_refresh_with_expired_token_returns_401() -> None:
    row = _active_token_row(expires_in_days=-1)  # scaduto ieri
    session = FakeSession(script=[row])
    client = _make_client(session)

    resp = client.post("/auth/refresh", json={"refresh_token": "expired-token"})

    assert resp.status_code == 401
    # Nessuna rotation deve avvenire su un token scaduto.
    assert len(session.executed_sql) == 1


def test_refresh_reuse_of_revoked_token_revokes_whole_family() -> None:
    """Un token già revocato ripresentato è evidenza di furto/riuso: l'intera
    family va revocata, non solo il singolo token, e la risposta resta 401."""
    family_id = uuid.uuid4()
    row = _active_token_row(family_id=family_id)
    row["revoked"] = True
    session = FakeSession(script=[row])
    client = _make_client(session)

    resp = client.post("/auth/refresh", json={"refresh_token": "stolen-token"})

    assert resp.status_code == 401
    assert session.committed is True
    # SELECT ... FOR UPDATE, poi UPDATE che revoca l'intera family.
    assert len(session.executed_sql) == 2
    revoke_sql = session.executed_sql[1]
    assert "WHERE family_id = :family_id" in revoke_sql
    assert "reuse_detected" not in revoke_sql  # passato come parametro, non nel testo SQL


def test_logout_all_revokes_every_family_for_the_user() -> None:
    user_id = uuid.uuid4()
    session = FakeSession(script=[{"user_id": user_id}])
    client = _make_client(session)

    resp = client.post("/auth/logout-all", json={"refresh_token": "some-token"})

    assert resp.status_code == 204
    assert len(session.executed_sql) == 2
    assert "WHERE user_id = :user_id AND NOT revoked" in session.executed_sql[1]


def test_logout_all_with_unknown_token_returns_401() -> None:
    session = FakeSession(script=[None])
    client = _make_client(session)

    resp = client.post("/auth/logout-all", json={"refresh_token": "unknown"})

    assert resp.status_code == 401
