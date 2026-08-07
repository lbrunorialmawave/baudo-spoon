"""End-to-end test for multi-column sort on ``GET /overview/players``.

The endpoint loads the hybrid artifact via ``_load_hybrid_results``; this
test replaces that call with a fixture payload, same pattern as
``test_mantra_players_endpoint.py``. ``get_artifact_store``/``get_repository``
read from ``app.state`` — set to plain sentinel objects since the mocked
``_load_hybrid_results`` never actually uses them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


def _players() -> list[dict[str, Any]]:
    """Two players share Pz1=10 (tie), one has a lower Pz1. Their
    expert_totale differs — the classic case for verifying a secondary
    sort key breaks ties left by the primary one."""
    return [
        {"fantacalcio_id": 1, "player_name": "Alpha", "Pz1": 10, "expert_totale": 30},
        {"fantacalcio_id": 2, "player_name": "Beta", "Pz1": 10, "expert_totale": 40},
        {"fantacalcio_id": 3, "player_name": "Gamma", "Pz1": 5, "expert_totale": 10},
    ]


def _payload() -> dict[str, Any]:
    return {"meta": {"season_start": 2025}, "players": _players()}


@pytest.fixture
def overview_client():
    """Test client wired to the ``/overview`` router, with the two extra
    app-state-backed dependencies (artifact store / repo) satisfied by
    sentinel objects — ``_load_hybrid_results`` is mocked per-test and
    never actually touches them."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import overview as overview_router

    app = FastAPI()
    app.include_router(overview_router.router)
    app.state.artifact_store = object()
    app.state.repo = object()
    return TestClient(app, raise_server_exceptions=False)


def _get(client, **params):
    with patch(
        "api.routers.overview._load_hybrid_results",
        return_value=(2025, _payload()),
    ):
        return client.get("/overview/players", params=params)


def test_single_sort_key_ascending(overview_client):
    """Regression: a single criterion still works exactly as before."""
    response = _get(overview_client, sort_by="Pz1")
    assert response.status_code == 200, response.text
    names = [p["playerName"] for p in response.json()["items"]]
    assert names == ["Gamma", "Alpha", "Beta"]  # Pz1: 5, 10, 10 (stable — original order among ties)


def test_single_sort_key_descending(overview_client):
    response = _get(overview_client, sort_by="-Pz1")
    assert response.status_code == 200
    names = [p["playerName"] for p in response.json()["items"]]
    assert names == ["Alpha", "Beta", "Gamma"]


def test_two_sort_keys_secondary_breaks_tie(overview_client):
    """Pz1 asc, then expert_totale desc: Gamma (Pz1=5) first, then among
    the Pz1=10 tie, Beta (expert_totale=40) before Alpha (30)."""
    response = _get(overview_client, sort_by="Pz1,-expert_totale")
    assert response.status_code == 200, response.text
    names = [p["playerName"] for p in response.json()["items"]]
    assert names == ["Gamma", "Beta", "Alpha"]


def test_two_sort_keys_secondary_ascending(overview_client):
    """Same primary key, secondary flipped to ascending: tie-break order reverses."""
    response = _get(overview_client, sort_by="Pz1,expert_totale")
    assert response.status_code == 200
    names = [p["playerName"] for p in response.json()["items"]]
    assert names == ["Gamma", "Alpha", "Beta"]


def test_unknown_sort_field_returns_422(overview_client):
    response = _get(overview_client, sort_by="NotAField")
    assert response.status_code == 422
    assert "Unknown sort field" in response.json()["detail"]


def test_duplicate_sort_field_returns_422(overview_client):
    response = _get(overview_client, sort_by="Pz1,-Pz1")
    assert response.status_code == 422
    assert "Duplicate sort field" in response.json()["detail"]


def test_too_many_sort_keys_returns_422(overview_client):
    response = _get(overview_client, sort_by="Pz1,VR,FP_Mantra,expert_totale")
    assert response.status_code == 422
    assert "Too many sort keys" in response.json()["detail"]


def test_no_sort_by_preserves_artifact_order(overview_client):
    """No sort_by → natural artifact order, unchanged (no accidental sort)."""
    response = _get(overview_client)
    assert response.status_code == 200
    names = [p["playerName"] for p in response.json()["items"]]
    assert names == ["Alpha", "Beta", "Gamma"]
