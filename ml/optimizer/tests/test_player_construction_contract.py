"""Contract: every non-default Player field has a DB construction path.

Mirrors ``_players_from_pool_rows`` without importing the FastAPI router
(so the test runs without the full API dependency stack).
"""

from __future__ import annotations

import dataclasses
from typing import cast

import pytest

from ml.optimizer.models import Player, Role


def _players_from_pool_rows(rows: list[dict]) -> list[Player]:
    """Copy of api.src.routers.optimizer._players_from_pool_rows — keep in sync."""
    return [
        Player(
            player_id=r["player_id"],
            name=r["name"],
            role=cast(Role, r["role"]),
            real_team=r["real_team"],
            cost=int(r["cost"]),
            projected_score=float(r["projected_score"]),
            prediction_std=r.get("prediction_std"),
            eligible_roles=frozenset(r.get("eligible_roles") or []),
            historical_overpay_ratio=r.get("historical_overpay_ratio"),
            season_value=r.get("season_value"),
            start_probability=r.get("start_probability"),
            reliability_weight=r.get("reliability_weight"),
            sample_cohort=r.get("sample_cohort"),
        )
        for r in rows
    ]


_ENRICHMENT_ONLY = frozenset({"var_score", "esv", "fp_ibrido"})


def _optional_fields(cls: type) -> list[str]:
    return [
        f.name
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING  # type: ignore[comparison-overlap]
    ]


def test_players_from_pool_rows_propagates_reliability_weight() -> None:
    rows = [
        {
            "player_id": "fm-1",
            "name": "Test",
            "role": "C",
            "real_team": "Roma",
            "cost": 15,
            "projected_score": 7.0,
            "prediction_std": 0.3,
            "season_value": 180.0,
            "start_probability": 0.8,
            "eligible_roles": [],
            "reliability_weight": 0.65,
            "sample_cohort": "LIMITED",
            "historical_overpay_ratio": 1.1,
        }
    ]
    players = _players_from_pool_rows(rows)
    assert len(players) == 1
    p = players[0]
    assert p.reliability_weight == pytest.approx(0.65)
    assert p.prediction_std == pytest.approx(0.3)
    assert p.season_value == pytest.approx(180.0)
    assert p.start_probability == pytest.approx(0.8)
    assert p.historical_overpay_ratio == pytest.approx(1.1)


def test_every_db_path_field_is_populated_from_pool_row() -> None:
    optional = set(_optional_fields(Player)) - _ENRICHMENT_ONLY
    row = {
        "player_id": "fm-sentinel",
        "name": "Sentinel",
        "role": "A",
        "real_team": "Inter",
        "cost": 20,
        "projected_score": 7.5,
        "prediction_std": 0.42,
        "historical_overpay_ratio": 1.23,
        "season_value": 210.0,
        "start_probability": 0.91,
        "eligible_roles": ["A"],
        "reliability_weight": 0.65,
        "sample_cohort": "LIMITED",
    }
    players = _players_from_pool_rows([row])
    p = players[0]

    missing: list[str] = []
    for field in sorted(optional):
        value = getattr(p, field)
        if field == "eligible_roles":
            if not value:
                missing.append(field)
            continue
        if value is None:
            missing.append(field)

    assert not missing, (
        f"_players_from_pool_rows does not populate: {missing}. "
        f"Add r.get('{missing[0]}') (or equivalent) to the Player(...) call."
    )


def test_reliability_weight_defaults_to_none_when_absent() -> None:
    rows = [
        {
            "player_id": "fm-2",
            "name": "Legacy",
            "role": "D",
            "real_team": "Milan",
            "cost": 12,
            "projected_score": 6.5,
        }
    ]
    players = _players_from_pool_rows(rows)
    assert players[0].reliability_weight is None
