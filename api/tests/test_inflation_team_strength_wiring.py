"""Router-level tests: ``team_strength_multiplier`` is settable from both APIs.

Process guard: these are entry-point reachability tests for the HTTP
layer — they prove that a value sent in the request schema actually
reaches :func:`estimate_effective_cost`, not just that the schema
accepts the field.  Mirrors the bar of
``ml/optimizer/tests/test_team_strength_wiring.py``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from api.routers import auction as auction_router
from api.routers import optimizer as optimizer_router
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from api.src.config import settings
from ml.optimizer.inflation import estimate_effective_cost
from ml.optimizer.models import InflationConfig, Player
from ml.optimizer.team_strength import load_team_strength_scores

# ---------------------------------------------------------------------------
# Auth helper: mint a valid member JWT against the dev secret.
# ---------------------------------------------------------------------------


def _member_token() -> str:
    """Issue a ``member`` JWT signed with the dev secret.

    Mirrors what ``/auth/login`` would return so the routers' auth
    dependency (``require_role("member")``) is satisfied in tests.
    """
    return jwt.encode(
        {"sub": "test-user", "role": "member"},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {_member_token()}"}


# ---------------------------------------------------------------------------
# Optimizer router
# ---------------------------------------------------------------------------


_optimizer_app = FastAPI()
_optimizer_app.include_router(optimizer_router.router)


@pytest.fixture
def optimizer_client() -> TestClient:
    return TestClient(_optimizer_app, raise_server_exceptions=False)


def _player(pid: str, team: str, role: str = "A", cost: int = 20, score: float = 8.5) -> Player:
    return Player(
        player_id=pid,
        name=pid.upper(),
        role=role,
        real_team=team,
        cost=cost,
        projected_score=score,
    )


def test_optimizer_router_propagates_team_strength_multiplier(
    optimizer_client: TestClient,
) -> None:
    """``inflationConfig.teamStrengthMultiplier`` reaches ``estimate_effective_cost``.

    Acceptance: building an ``OptimizationRequest`` with the field set
    > 0 must produce an ``OptimizationConfig`` whose
    ``inflation_config.team_strength_multiplier`` carries the same value,
    and that value must actually change the cost returned by
    :func:`estimate_effective_cost` for a player on a strong team.
    """
    from api.src.schemas import (
        FormationSchema,
        InflationConfigSchema,
        OptimizationRequest,
    )

    req = OptimizationRequest(
        season_start=2025,
        budget=500,
        formations=[FormationSchema(label="3-4-3", defenders=3, midfielders=4, forwards=3)],
        num_participants=8,
        max_players_per_team=6,
        big_teams_cap=25,
        min_distinct_teams=2,
        ruleset="classic",
        inflation_config=InflationConfigSchema(team_strength_multiplier=1.5),
    )

    config = optimizer_router._build_config(req)  # test the wire

    # 1. Field carried through to the dataclass.
    assert config.inflation_config.team_strength_multiplier == pytest.approx(1.5)

    # 2. Value actually changes the computed effective cost.
    p = _player("p1", "Inter")
    ts_scores = load_team_strength_scores(known_teams={"Inter", "Lecce"})

    eff_off = estimate_effective_cost(
        player=p,
        role_percentile=0.9,
        num_participants=8,
        config=InflationConfig(),
        team_strength_scores=ts_scores,
    )
    eff_on = estimate_effective_cost(
        player=p,
        role_percentile=0.9,
        num_participants=8,
        config=config.inflation_config,
        team_strength_scores=ts_scores,
    )
    assert eff_on != eff_off
    assert eff_on > eff_off  # Inter has high Elo → cost must increase.


def test_optimizer_router_default_team_strength_multiplier_is_zero(
    optimizer_client: TestClient,
) -> None:
    """No-``teamStrengthMultiplier`` request → multiplier defaults to ``0.0``.

    Backward-compat: requests that don't set the field must still
    produce an ``InflationConfig`` with ``team_strength_multiplier=0.0``,
    matching the dataclass default.  This guards the schema default and
    the ``_build_config`` wire against silent regressions.
    """
    from api.src.schemas import OptimizationRequest

    req = OptimizationRequest(season_start=2025)
    config = optimizer_router._build_config(req)
    assert config.inflation_config.team_strength_multiplier == 0.0


# ---------------------------------------------------------------------------
# Auction router
# ---------------------------------------------------------------------------


_auction_app = FastAPI()
_auction_app.include_router(auction_router.router)


@pytest.fixture
def auction_client() -> TestClient:
    return TestClient(_auction_app, raise_server_exceptions=False)


def _participant(pid: str, budget: int = 500) -> dict[str, object]:
    return {
        "participantId": pid,
        "displayName": pid,
        "budgetInitial": budget,
    }


def _auction_pool() -> list[dict[str, object]]:
    return [
        {
            "playerId": "p1",
            "name": "P1",
            "role": "A",
            "realTeam": "Inter",
            "cost": 20,
            "projectedScore": 8.5,
        },
    ]


def test_auction_router_propagates_team_strength_multiplier(
    auction_client: TestClient,
) -> None:
    """``inflationConfig.teamStrengthMultiplier`` reaches ``estimate_effective_cost``.

    Acceptance: ``POST /auction/init`` with the field set > 0 must
    produce an :class:`AuctionConfig` whose
    ``inflation_config.team_strength_multiplier`` carries the same value,
    and that value must actually change the cost returned by
    :func:`estimate_effective_cost`.
    """
    payload = cast(
        dict[str, Any],
        {
            "seasonStart": 2025,
            "participants": [_participant("u1"), _participant("u2")],
            "config": {
                "numParticipants": 2,
                "useInflationBaseline": True,
                "inflationConfig": {
                    "teamStrengthMultiplier": 1.5,
                },
            },
            "playerPool": _auction_pool(),
        },
    )

    resp = auction_client.post(
        "/auction/init", json=payload, headers=_auth_header()
    )
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["sessionId"]

    # 1. Field carried through to the constructed AuctionConfig.
    store = getattr(auction_client.app.state, "auction_sessions", None)
    assert store is not None
    session = store[session_id]
    assert session.state.config.inflation_config is not None
    assert session.state.config.inflation_config.team_strength_multiplier == pytest.approx(1.5)

    # 2. Value actually changes the computed effective cost.
    p = session.state.available_pool[0]
    ts_scores = load_team_strength_scores(known_teams={"Inter", "Lecce"})

    eff_off = estimate_effective_cost(
        player=p,
        role_percentile=0.9,
        num_participants=8,
        config=InflationConfig(),
        team_strength_scores=ts_scores,
    )
    eff_on = estimate_effective_cost(
        player=p,
        role_percentile=0.9,
        num_participants=8,
        config=session.state.config.inflation_config,
        team_strength_scores=ts_scores,
    )
    assert eff_on != eff_off
    assert eff_on > eff_off  # Inter has high Elo → cost must increase.
