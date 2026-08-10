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


def test_no_fvm_fallback_when_no_prediction(repo: DataRepository):
    """FVM must not be used as projected_score; player without ML is excluded."""
    pq = _make_pq(2, "DEF", 15, 6.0)  # fvm=6.0 would previously become projected_score
    pim = MagicMock()
    pim.fantacalcio_id = 2
    pim.player_fotmob_id = None
    pim.name_fotmob = None
    pim.team_fotmob = None
    row = (pq, pim)

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=[]):
        pool, excluded = _run(
            repo.get_player_pool(db, season_start=2025, return_exclusions=True)
        )

    assert pool == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "no_projection"
    assert excluded[0]["player_id"] == "fc-2"


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


def test_excluded_no_projection_is_observable(repo: DataRepository):
    """Players with valid cost but no projection appear in excluded list, not silently dropped."""
    pq = _make_pq(99, "FWD", 20, None)
    pq.fvm = None
    pim = MagicMock()
    pim.fantacalcio_id = 99
    pim.player_fotmob_id = None
    pim.name_fotmob = None
    pim.team_fotmob = None
    row = (pq, pim)

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=[]):
        pool, excluded = _run(
            repo.get_player_pool(db, season_start=2025, return_exclusions=True)
        )

    assert pool == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "no_projection"
    assert excluded[0]["player_id"] == "fc-99"
    assert excluded[0]["cost"] == 20


def test_fvm_out_of_scale_not_used_as_projected_score(repo: DataRepository):
    """A player with high FVM (e.g. 17) and no ML prediction must be excluded,
    not enter the pool with projected_score=17.0."""
    pq = _make_pq(50, "FWD", 25, 17.2)
    pim = MagicMock()
    pim.fantacalcio_id = 50
    pim.player_fotmob_id = None
    pim.name_fotmob = None
    pim.team_fotmob = None
    row = (pq, pim)

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=[]):
        pool, excluded = _run(
            repo.get_player_pool(db, season_start=2025, return_exclusions=True)
        )

    assert pool == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "no_projection"
    assert excluded[0]["player_id"] == "fc-50"


def test_ml_prediction_still_used_when_present(repo: DataRepository):
    """Regression: valid ML predicted_fantavoto continues to populate the pool."""
    pq = _make_pq(10, "MID", 18, 17.2)  # high FVM must be ignored when ML exists
    pim = _make_pim(10, 1010)
    row = (pq, pim)

    predictions = [
        {
            "player_fotmob_id": 1010,
            "predicted_fantavoto": 6.8,
            "fantavoto_medio": 6.5,
            "expected_minutes": 2000.0,
            "prediction_std": 0.4,
        }
    ]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=predictions):
        pool = _run(repo.get_player_pool(db, season_start=2025))

    assert len(pool) == 1
    assert pool[0]["projected_score"] == pytest.approx(6.8)


def test_implausible_ml_score_is_excluded(repo: DataRepository):
    """ML predicted_fantavoto outside the plausible range is treated as no projection."""
    pq = _make_pq(11, "FWD", 22, 8.0)
    pim = _make_pim(11, 1111)
    row = (pq, pim)

    predictions = [
        {
            "player_fotmob_id": 1111,
            "predicted_fantavoto": 17.2,  # out of scale
            "fantavoto_medio": 6.5,
            "expected_minutes": 2000.0,
            "prediction_std": 0.4,
        }
    ]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    with patch.object(repo, "get_predictions", new_callable=AsyncMock, return_value=predictions):
        pool, excluded = _run(
            repo.get_player_pool(db, season_start=2025, return_exclusions=True)
        )

    assert pool == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "implausible_projection"
    assert excluded[0]["player_id"] == "fm-1111"
