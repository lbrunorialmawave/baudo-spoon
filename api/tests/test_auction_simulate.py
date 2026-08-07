"""Tests for POST /auction/simulate (stateless Monte Carlo)."""
from __future__ import annotations
from typing import Any, cast
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from api.routers import auction as auction_router
from api.src.config import settings
from api.src.deps import get_db

def _member_token() -> str:
    return jwt.encode({"sub": "test-user", "role": "member"}, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {_member_token()}"}

async def _noop_db():
    yield None

def _player(pid, role, cost, score):
    return {"playerId": pid, "name": pid.upper(), "role": role, "realTeam": "TEST", "cost": cost, "projectedScore": score}

def _participant(pid, budget=300):
    return {"participantId": pid, "displayName": pid, "budgetInitial": budget}

def _profile(pid, aggressiveness=0.5):
    return {"participantId": pid, "policy": {"aggressiveness": aggressiveness, "maxOverpayRatio": 1.2, "minResidualCreditsPerSlot": 1.0}}

def _pool():
    players = []
    for i in range(4): players.append(_player(f"p{i}", "P", 5+i, 5.0+i*0.2))
    for i in range(6): players.append(_player(f"d{i}", "D", 4+i, 5.0+i*0.15))
    for i in range(6): players.append(_player(f"c{i}", "C", 6+i, 5.5+i*0.15))
    for i in range(4): players.append(_player(f"a{i}", "A", 8+i, 6.0+i*0.2))
    return players

def _valid_payload():
    return {
        "seasonStart": 2025,
        "participants": [_participant("u0"), _participant("u1")],
        "config": {"numParticipants": 2, "roleQuotas": {"P": 1, "D": 2, "C": 2, "A": 1}, "budgetInitial": 300, "referenceBudget": 300},
        "playerPool": _pool(),
        "bidderProfiles": [_profile("u0", 0.4), _profile("u1", 0.6)],
        "simConfig": {"nSimulations": 5, "randomSeed": 42},
    }

@pytest.fixture
def app():
    application = FastAPI()
    application.include_router(auction_router.router)
    application.dependency_overrides[get_db] = _noop_db
    return application

@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)

def test_simulate_ok_shape(client):
    resp = client.post("/auction/simulate", json=_valid_payload(), headers=_auth_header())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["nCompleted"] == 5
    assert "u0" in body["perParticipant"]
    assert "spendP50" in body["perParticipant"]["u0"]

def test_simulate_n_simulations_out_of_range(client):
    payload = _valid_payload()
    sim = cast(dict[str, Any], payload["simConfig"])
    sim["nSimulations"] = 0
    assert client.post("/auction/simulate", json=payload, headers=_auth_header()).status_code == 422
    sim["nSimulations"] = 501
    assert client.post("/auction/simulate", json=payload, headers=_auth_header()).status_code == 422

def test_simulate_does_not_touch_session_store(client, app):
    app.state.auction_sessions = {"seed": object()}
    before = set(app.state.auction_sessions.keys())
    resp = client.post("/auction/simulate", json=_valid_payload(), headers=_auth_header())
    assert resp.status_code == 200, resp.text
    assert set(getattr(app.state, "auction_sessions", {}).keys()) == before
