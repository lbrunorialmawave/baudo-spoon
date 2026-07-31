"""End-to-end test for ``GET /mantra/players``.

Acceptance gate for the P1-4 scope-boundary plumbing: the
``season_value`` / ``start_probability`` fields must be present on every
item in the JSON payload served by the endpoint, exactly as the
MANTRA runner wrote them into ``mantra_results_{season}.json``.

The endpoint reads the JSON file from disk via ``_load_mantra_results``;
this test replaces that call with a fixture payload so the test is
fully decoupled from the DB and the MANTRA compute pipeline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


# Minimal payload mirroring the MANTRA runner's output shape. The two
# fields under test are deliberately the last to be set so the
# assertions catch any regression in the projection step.
def _mantra_payload() -> dict[str, Any]:
    return {
        "meta": {
            "season_start": 2025,
            "n_players": 3,
        },
        "players": [
            {
                "fantacalcio_id": 1,
                "player_fotmob_id": 100,
                "player_name": "Alpha",
                "team": "AAA",
                "ruolo_primario": "FWD",
                "FP_Mantra": 7.5,
                "VR": 9.0,
                "Prezzo_Massimo": 15.0,
                "Fase7": "TOP",
                "rischio": "low",
                "season_value": 210.0,
                "start_probability": 0.79,
            },
            {
                "fantacalcio_id": 2,
                "player_fotmob_id": 200,
                "player_name": "Beta",
                "team": "BBB",
                "ruolo_primario": "MID",
                "FP_Mantra": 6.0,
                "VR": 5.0,
                "Prezzo_Massimo": 7.0,
                "Fase7": "AFFARE",
                "rischio": "high",
                # Older artefact, no ML prediction → both None.
                "season_value": None,
                "start_probability": None,
            },
            {
                "fantacalcio_id": 3,
                "player_fotmob_id": 300,
                "player_name": "Gamma",
                "team": "CCC",
                "ruolo_primario": "A",
                "FP_Mantra": 5.5,
                "VR": 4.0,
                "Prezzo_Massimo": 4.0,
                "Fase7": "RISCHIO",
                "rischio": "high",
                "season_value": 130.0,
                "start_probability": 0.5,
            },
        ],
        "classifications": {
            "top_per_ruolo": {},
            "multi_eleggibilita": {},
            "low_cost": [],
            "low_cost_titolari": [],
            "scommesse_multi_ruolo": [],
            "watchlist_giovani": [],
        },
    }


@pytest.fixture
def mantra_client():
    """Test client wired only to the ``/mantra`` router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import mantra as mantra_router

    app = FastAPI()
    app.include_router(mantra_router.router)
    return TestClient(app, raise_server_exceptions=False)


def test_mantra_players_endpoint_serves_season_value_and_start_probability(
    mantra_client,
):
    """Acceptance: the two fields are present in the JSON payload."""
    payload = _mantra_payload()
    with patch("api.routers.mantra._load_mantra_results", return_value=payload):
        response = mantra_client.get("/mantra/players")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 3
    assert body["size"] >= 3
    items = body["items"]
    assert len(items) == 3

    # Every record carries both new keys (no silent absence).
    for item in items:
        assert "season_value" in item
        assert "start_probability" in item

    alpha, beta, gamma = items

    # Alpha: matched prediction → both fields non-null and equal the runner's output.
    assert alpha["player_fotmob_id"] == 100
    assert alpha["season_value"] == pytest.approx(210.0)
    assert alpha["start_probability"] == pytest.approx(0.79)

    # Beta: missing prediction → both None, no crash, no leak.
    assert beta["player_fotmob_id"] == 200
    assert beta["season_value"] is None
    assert beta["start_probability"] is None

    # Gamma: matched prediction with different values.
    assert gamma["player_fotmob_id"] == 300
    assert gamma["season_value"] == pytest.approx(130.0)
    assert gamma["start_probability"] == pytest.approx(0.5)


def test_mantra_players_endpoint_passes_through_filters_with_new_fields(
    mantra_client,
):
    """Filters (e.g. ``ruolo``) must keep the new fields on the response."""
    payload = _mantra_payload()
    with patch("api.routers.mantra._load_mantra_results", return_value=payload):
        response = mantra_client.get("/mantra/players", params={"ruolo": "FWD"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    only = body["items"][0]
    assert only["player_name"] == "Alpha"
    assert only["season_value"] == pytest.approx(210.0)
    assert only["start_probability"] == pytest.approx(0.79)
