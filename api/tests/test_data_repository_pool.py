"""Unit tests for DataRepository.get_player_pool() — season_value fields."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.data_repository import DataRepository


def _make_pq(fantacalcio_id: int, role: str, qt_a: int, fvm: float, team: str = "Roma"):
    """Minimal PlayerQuotation-like object."""
    pq = MagicMock()
    pq.fantacalcio_id = fantacalcio_id
    pq.role = role
    pq.qt_a = qt_a
    pq.fvm = fvm
    pq.team = team
    pq.player_name = f"Player{fantacalcio_id}"
    pq.season_start = 2025
    return pq


def _make_pim(fantacalcio_id: int, fotmob_id: int):
    """Minimal PlayerIdMap-like object."""
    pim = MagicMock()
    pim.fantacalcio_id = fantacalcio_id
    pim.player_fotmob_id = fotmob_id
    pim.name_fotmob = f"FotMob{fotmob_id}"
    pim.team_fotmob = "Roma"
    pim.season_start = 2025
    return pim


@pytest.fixture
def repo() -> DataRepository:
    return DataRepository(artifacts_dir=Path("/tmp/test"))


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_season_value_fields_present_with_expected_minutes(repo: DataRepository):
    """When prediction has expected_minutes, season_value and start_probability are derived."""
    pq = _make_pq(1, "GK", 10, 6.5)
    pim = _make_pim(1, 100)
    row = (pq, pim)

    predictions = [
        {
            "player_fotmob_id": 100,
            "predicted_fantavoto": 7.0,
            "fantavoto_medio": 6.5,
            "expected_minutes": 2700.0,
            "prediction_std": 0.5,
        }
    ]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=predictions):
        pool = _run(repo.get_player_pool(db, season_start=2025))

    assert len(pool) == 1
    p = pool[0]
    assert p["projected_score"] == pytest.approx(7.0)
    assert p["season_value"] == pytest.approx(210.0)
    assert p["start_probability"] == pytest.approx(2700.0 / 3420.0)


def test_season_value_none_when_no_prediction(repo: DataRepository):
    """When no ML prediction is available, season_value/start_probability are None."""
    pq = _make_pq(2, "DEF", 15, 6.0)
    pim = MagicMock()
    pim.fantacalcio_id = 2
    pim.player_fotmob_id = None
    pim.name_fotmob = None
    pim.team_fotmob = None
    row = (pq, pim)

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=[]):
        pool = _run(repo.get_player_pool(db, season_start=2025))

    assert len(pool) == 1
    p = pool[0]
    assert p["projected_score"] == pytest.approx(6.0)
    assert p["season_value"] is None
    assert p["start_probability"] is None


def test_season_value_prefers_artifact_fantapunti_totali(repo: DataRepository):
    """When prediction has fantapunti_totali directly, use it over derivation."""
    pq = _make_pq(3, "MID", 20, 7.0)
    pim = _make_pim(3, 300)
    row = (pq, pim)

    predictions = [
        {
            "player_fotmob_id": 300,
            "predicted_fantavoto": 7.0,
            "fantavoto_medio": 6.8,
            "expected_minutes": 2700.0,
            "prediction_std": 0.3,
            "fantapunti_totali": 200.0,
            "probabilita_titolarita": 0.85,
        }
    ]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=predictions):
        pool = _run(repo.get_player_pool(db, season_start=2025))

    assert len(pool) == 1
    p = pool[0]
    assert p["season_value"] == pytest.approx(200.0)
    assert p["start_probability"] == pytest.approx(0.85)


def test_all_pool_entries_have_season_value_keys(repo: DataRepository):
    """Every returned dict must have season_value and start_probability keys."""
    rows = []
    predictions = []
    for i in range(1, 4):
        pq = _make_pq(i, "FWD", 10 + i, 6.0 + i * 0.2)
        pim = _make_pim(i, 1000 + i)
        rows.append((pq, pim))
        predictions.append(
            {
                "player_fotmob_id": 1000 + i,
                "predicted_fantavoto": 6.0 + i * 0.2,
                "fantavoto_medio": 6.0,
                "expected_minutes": 1800.0 if i % 2 == 0 else 0.0,
                "prediction_std": 0.1,
            }
        )

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=rows)))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=predictions):
        pool = _run(repo.get_player_pool(db, season_start=2025))

    for p in pool:
        assert "season_value" in p
        assert "start_probability" in p
