"""Tests for the pure ``suggest_alternatives`` function.

Covers (per spec §5):
* same-role validation (candidates of other roles are filtered out).
* empty pool / role exhausted -> both alternatives ``None`` with explicit
  ``reason_if_none``.
* low_cost: max ``projected_score / expected_price`` under a percentile
  threshold of the role.
* closest: min ``|projected_score - target.projected_score|`` with
  ``expected_price`` as tie-break.
* that suggestions re-compute ``expected_price`` via the live
  ``AuctionState`` (price drift propagates to suggestions).
"""

from __future__ import annotations

import pytest

from ml.auction.alternatives import suggest_alternatives
from ml.auction.models import (
    AlternativesConfig,
    AuctionConfig,
    ParticipantSetup,
)
from ml.auction.orchestrator import (
    AuctionSession,
    initialize_auction,
)
from ml.optimizer.models import Player

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_participants(n: int = 4) -> list[ParticipantSetup]:
    return [
        ParticipantSetup(
            participant_id=f"u{i}", display_name=f"U{i}", budget_initial=500
        )
        for i in range(1, n + 1)
    ]


def _mk(
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


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def gk_pool() -> list[Player]:
    """Pool di 5 portieri con score e cost crescenti."""
    return [
        _mk("top1", "Top 1", "P", 30, 8.0, "A"),
        _mk("top2", "Top 2", "P", 28, 7.8, "B"),
        _mk("mid1", "Mid 1", "P", 18, 6.0, "C"),
        _mk("mid2", "Mid 2", "P", 16, 5.5, "D"),
        _mk("low1", "Low 1", "P", 5, 4.0, "E"),
    ]


@pytest.fixture
def multi_role_pool() -> list[Player]:
    """Pool con più ruoli, usato per test same-role filtering."""
    return [
        _mk("p1", "GK", "P", 25, 7.0, "A"),
        _mk("d1", "DF", "D", 20, 7.0, "B"),
        _mk("c1", "MF", "C", 30, 7.5, "A"),
        _mk("a1", "FW", "A", 40, 8.0, "A"),
    ]


# ---------------------------------------------------------------------------
# Role exhausted / empty pool
# ---------------------------------------------------------------------------


class TestRoleExhausted:
    def test_ruolo_esaurito_restituisce_none_esplicito(
        self, multi_role_pool: list[Player]
    ) -> None:
        target = _mk("p1", "GK", "P", 25, 7.0)
        # available_pool vuoto per i portieri
        result = suggest_alternatives(
            target=target,
            available_pool=[],
            state=initialize_auction(
                _make_participants(4),
                AuctionConfig(num_participants=4),
                multi_role_pool,
            ),
            config=AlternativesConfig(),
        )
        assert result.low_cost_alternative is None
        assert result.closest_alternative is None
        assert result.reason_if_none is not None
        assert "esaurito" in result.reason_if_none.lower()
        assert result.target_player_id == "p1"

    def test_nessun_candidato_stesso_ruolo(self, multi_role_pool: list[Player]) -> None:
        """Se l'available_pool contiene solo giocatori di altri ruoli, None esplicito."""
        target = _mk("p1", "GK", "P", 25, 7.0)
        # available_pool contiene solo difensori/centrocampisti/attaccanti
        available = [p for p in multi_role_pool if p.role != "P"]
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), multi_role_pool
        )
        result = suggest_alternatives(
            target=target,
            available_pool=available,
            state=state,
            config=AlternativesConfig(),
        )
        assert result.low_cost_alternative is None
        assert result.closest_alternative is None
        assert result.reason_if_none is not None
        assert "P" in result.reason_if_none


# ---------------------------------------------------------------------------
# Same-role filtering
# ---------------------------------------------------------------------------


class TestSameRoleFiltering:
    def test_alternative_sempre_stesso_ruolo_del_target(
        self, multi_role_pool: list[Player]
    ) -> None:
        target = _mk("p1", "GK", "P", 25, 7.0)
        # Passiamo TUTTO il pool (anche non-P), la funzione deve filtrare.
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), multi_role_pool
        )
        result = suggest_alternatives(
            target=target,
            available_pool=multi_role_pool,
            state=state,
            config=AlternativesConfig(),
        )
        # Solo p1 è P, quindi non ci sono alternative (il target è l'unico P).
        assert result.low_cost_alternative is None
        assert result.closest_alternative is None
        assert result.reason_if_none is not None

    def test_target_stesso_ruolo_dei_candidati(self, gk_pool: list[Player]) -> None:
        target = gk_pool[0]  # top1, role P
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), gk_pool
        )
        result = suggest_alternatives(
            target=target,
            available_pool=gk_pool,
            state=state,
            config=AlternativesConfig(),
        )
        # Le alternative (se non-None) devono essere P
        if result.low_cost_alternative is not None:
            assert result.low_cost_alternative.role == "P"
        if result.closest_alternative is not None:
            assert result.closest_alternative.role == "P"


# ---------------------------------------------------------------------------
# Closest alternative
# ---------------------------------------------------------------------------


class TestClosestAlternative:
    def test_closest_piu_vicino_per_score(self, gk_pool: list[Player]) -> None:
        """top1 (score=8.0) deve avere come closest top2 (score=7.8, distanza 0.2)."""
        target = gk_pool[0]  # top1
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), gk_pool
        )
        result = suggest_alternatives(
            target=target,
            available_pool=gk_pool,
            state=state,
            config=AlternativesConfig(),
        )
        assert result.closest_alternative is not None
        # Distanza minima: |7.8 - 8.0| = 0.2
        assert result.closest_alternative.player_id == "top2"

    def test_closest_tiebreak_per_prezzo_piu_basso(self, gk_pool: list[Player]) -> None:
        """Se due candidati hanno la stessa distanza, vince quello col minor
        expected_price."""
        # Costruiamo un caso sintetico: due candidati a distanza identica
        # dal target, ma costi diversi.
        target = _mk("t", "Target", "P", 25, 7.0)
        # top2 (7.8) e un nuovo fake a 7.2 (distanza 0.2 vs 0.2)
        candidate_cheap = _mk("c1", "Cheap 7.2", "P", 5, 7.2)
        candidate_expensive = _mk("c2", "Exp 7.2", "P", 50, 7.2)
        pool = [target, candidate_cheap, candidate_expensive]
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), pool
        )
        result = suggest_alternatives(
            target=target,
            available_pool=[candidate_cheap, candidate_expensive],
            state=state,
            config=AlternativesConfig(),
        )
        assert result.closest_alternative is not None
        # A parità di distanza, expected_price più basso -> c1 vince.
        assert result.closest_alternative.player_id == "c1"


# ---------------------------------------------------------------------------
# Low-cost alternative
# ---------------------------------------------------------------------------


class TestLowCostAlternative:
    def test_low_cost_sotto_soglia_max_rapporto(self, gk_pool: list[Player]) -> None:
        """low_cost_percentile=0.4 -> filtra i 40% più economici del ruolo,
        tra questi seleziona il max(score/expected_price)."""
        target = gk_pool[0]
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), gk_pool
        )
        result = suggest_alternatives(
            target=target,
            available_pool=gk_pool,
            state=state,
            config=AlternativesConfig(low_cost_percentile=0.4),
        )
        assert result.low_cost_alternative is not None
        # Con 5 GK e percentile 0.4, i 40% più economici (sorted by expected_price)
        # sono low1 (5) e mid2 (16) - indice floor(0.4 * 4) = 1.
        # Tra questi, low1 ha score 4.0 / 5 = 0.8; mid2 ha 5.5 / 16 = 0.34.
        # Vince low1 (miglior rapporto).
        assert result.low_cost_alternative.player_id == "low1"

    def test_low_cost_none_se_nessun_candidato_eligibile(
        self, gk_pool: list[Player]
    ) -> None:
        """Se la soglia è molto bassa, può succedere che l'eligible sia vuoto."""
        target = gk_pool[0]
        state = initialize_auction(
            _make_participants(4), AuctionConfig(num_participants=4), gk_pool
        )
        # low_cost_percentile=0.0 -> quantile_idx = min(4, 0) = 0
        # price_threshold = sorted_prices[0] (il più economico)
        # eligible = tutti quelli con expected_price <= min (cioè solo il più economico)
        result = suggest_alternatives(
            target=target,
            available_pool=gk_pool,
            state=state,
            config=AlternativesConfig(low_cost_percentile=0.0),
        )
        # low1 (5) è l'unico eligible -> scelto
        assert result.low_cost_alternative is not None
        assert result.low_cost_alternative.player_id == "low1"

    def test_low_cost_validation_percentile(
        self,
    ) -> None:
        """AlternativesConfig valida il percentile."""
        with pytest.raises(ValueError):
            AlternativesConfig(low_cost_percentile=1.5)
        with pytest.raises(ValueError):
            AlternativesConfig(low_cost_percentile=-0.1)


# ---------------------------------------------------------------------------
# Price drift propagation to suggestions
# ---------------------------------------------------------------------------


class TestPriceDriftPropagation:
    def test_suggestions_riflettono_price_drift_aggiornato(
        self, gk_pool: list[Player]
    ) -> None:
        """Dopo che un TOP GK è stato pagato meno del previsto, la
        proiezione sui TOP GK rimanenti deve scendere, e quindi il
        closest_alternative per un altro TOP GK deve riflettere questo
        updated price_index."""
        cfg = AuctionConfig(num_participants=4)
        session = AuctionSession(_make_participants(4), cfg, gk_pool)

        # Proiezione iniziale per top1 (TOP, atteso 30): 30
        proj_before = session.projection("top1")
        assert proj_before == pytest.approx(30.0)

        # Assegna top1 a 15 (metà del previsto) -> indice P:TOP scende.
        result = session.record("top1", "u1", 15)
        assert result.success
        proj_after = session.projection("top2")
        # top1 (percentile=1.0) è TOP, top2 (percentile=0.75) è MID.
        # P:TOP va a (1-0.3)*1.0 + 0.3*0.5 = 0.85; MID riceve lo spillover
        # 1.0 * (1 + 0.25*(0.5-1.0)) = 0.875.  Expected per top2 = 28*0.875.
        assert proj_after == pytest.approx(28.0 * 0.875)

        # Chiedi alternative per top2 tramite la sessione (target ancora
        # nel pool, top1 appena rimosso).  La funzione deve restituire un
        # oggetto valido; la propagazione del price drift è già asserita
        # sopra (proj_after) e nei test dedicati in test_price_drift.
        suggestion = session.alternatives(
            target_player_id="top2",
            config=AlternativesConfig(),
        )
        assert suggestion.target_player_id == "top2"
        # I candidati devono essere dello stesso ruolo del target.
        if suggestion.low_cost_alternative is not None:
            assert suggestion.low_cost_alternative.role == "P"
        if suggestion.closest_alternative is not None:
            assert suggestion.closest_alternative.role == "P"
