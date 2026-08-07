"""Shared pytest fixtures for the live auction tests."""

from __future__ import annotations

import pytest

from ml.auction.models import (
    AlternativesConfig,
    AuctionConfig,
    MarketDriftConfig,
    ParticipantSetup,
)
from ml.optimizer.models import Player

# ---------------------------------------------------------------------------
# Player fixtures
# ---------------------------------------------------------------------------


def _mk_player(
    pid: str,
    name: str,
    role: str,
    cost: int,
    score: float,
    team: str = "TEST",
) -> Player:
    return Player(
        player_id=pid,
        name=name,
        real_team=team,
        role=role,  # type: ignore[arg-type]
        cost=cost,
        projected_score=score,
    )


@pytest.fixture
def goalkeeper_pool() -> list[Player]:
    """Pool di portieri con score distribuiti tra LOW/MID/TOP."""
    return [
        _mk_player("p_top1", "Top GK 1", "P", 30, 8.0, "A"),
        _mk_player("p_top2", "Top GK 2", "P", 28, 7.8, "B"),
        _mk_player("p_mid1", "Mid GK 1", "P", 20, 6.0, "C"),
        _mk_player("p_mid2", "Mid GK 2", "P", 18, 5.5, "D"),
        _mk_player("p_low1", "Low GK 1", "P", 8, 4.0, "E"),
        _mk_player("p_low2", "Low GK 2", "P", 5, 3.5, "F"),
    ]


@pytest.fixture
def mixed_pool() -> list[Player]:
    """Pool misto con tutti i ruoli (per i test sull'orchestratore)."""
    return [
        # Portieri
        _mk_player("p1", "GK Star", "P", 30, 8.0, "BIG1"),
        _mk_player("p2", "GK Mid", "P", 18, 6.0, "MID1"),
        _mk_player("p3", "GK Low", "P", 5, 4.0, "SMALL1"),
        # Difensori
        _mk_player("d1", "DF Star", "D", 25, 7.5, "BIG1"),
        _mk_player("d2", "DF Good", "D", 15, 6.0, "MID1"),
        _mk_player("d3", "DF Low", "D", 4, 4.5, "SMALL1"),
        # Centrocampisti
        _mk_player("c1", "MF Star", "C", 35, 8.5, "BIG1"),
        _mk_player("c2", "MF Good", "C", 18, 6.5, "MID1"),
        _mk_player("c3", "MF Low", "C", 6, 5.0, "SMALL1"),
        # Attaccanti
        _mk_player("a1", "FW Star", "A", 40, 9.0, "BIG1"),
        _mk_player("a2", "FW Good", "A", 22, 7.0, "MID1"),
        _mk_player("a3", "FW Low", "A", 7, 5.0, "SMALL1"),
    ]


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def market_drift_config() -> MarketDriftConfig:
    """Configurazione di mercato di default per i test."""
    return MarketDriftConfig()


@pytest.fixture
def aggressive_market_drift() -> MarketDriftConfig:
    """Configurazione con alpha alto e spillover cross-role attivo."""
    return MarketDriftConfig(
        alpha=0.5,
        spillover_adjacent_tier=0.4,
        spillover_cross_role=0.1,
    )


@pytest.fixture
def alternatives_config() -> AlternativesConfig:
    return AlternativesConfig()


@pytest.fixture
def auction_config(alternatives_config: AlternativesConfig) -> AuctionConfig:
    return AuctionConfig(
        num_participants=4,
        alternatives_config=alternatives_config,
    )


@pytest.fixture
def participants() -> list[ParticipantSetup]:
    return [
        ParticipantSetup(
            participant_id=f"u{i}", display_name=f"User {i}", budget_initial=500
        )
        for i in range(1, 5)
    ]
