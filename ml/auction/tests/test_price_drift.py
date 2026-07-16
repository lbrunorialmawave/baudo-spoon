"""Tests for the pure price drift logic."""

from __future__ import annotations

import pytest

from ml.auction.models import (
    ALL_TIERS,
    AuctionConfig,
    MarketDriftConfig,
    ParticipantSetup,
    Role,
    Tier,
)
from ml.auction.orchestrator import (
    initialize_auction,
    record_assignment,
)
from ml.auction.price_drift import (
    build_initial_price_index,
    classify_tier,
    clamp_index,
    compute_baseline_cost,
    compute_expected_price,
    get_current_projection,
    update_price_index,
)
from ml.optimizer.inflation import InflationConfig
from ml.optimizer.models import Player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_participants(n: int = 4) -> list[ParticipantSetup]:
    return [
        ParticipantSetup(participant_id=f"u{i}", display_name=f"U{i}", budget_initial=500)
        for i in range(1, n + 1)
    ]


def _build_index_for_update() -> dict[Role, dict[Tier, float]]:
    return {
        "P": {"LOW": 1.0, "MID": 1.0, "TOP": 1.0},
        "D": {"LOW": 1.0, "MID": 1.0, "TOP": 1.0},
    }


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


class TestClassifyTier:
    def test_top_when_above_upper_threshold(self) -> None:
        cfg = MarketDriftConfig()
        assert classify_tier(0.95, cfg) == "TOP"
        assert classify_tier(0.80, cfg) == "TOP"  # limite incluso

    def test_mid_between_thresholds(self) -> None:
        cfg = MarketDriftConfig()
        assert classify_tier(0.79, cfg) == "MID"
        assert classify_tier(0.40, cfg) == "MID"  # limite incluso

    def test_low_below_lower_threshold(self) -> None:
        cfg = MarketDriftConfig()
        assert classify_tier(0.39, cfg) == "LOW"
        assert classify_tier(0.0, cfg) == "LOW"

    def test_custom_thresholds(self) -> None:
        cfg = MarketDriftConfig(tier_thresholds=(0.2, 0.6))
        assert classify_tier(0.65, cfg) == "TOP"
        assert classify_tier(0.5, cfg) == "MID"
        assert classify_tier(0.1, cfg) == "LOW"

    def test_percentile_out_of_range_clamped(self) -> None:
        cfg = MarketDriftConfig()
        assert classify_tier(1.05, cfg) == "TOP"  # clampato a 1.0
        assert classify_tier(-0.1, cfg) == "LOW"  # clampato a 0.0

    def test_invalid_config_thresholds(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(tier_thresholds=(0.8, 0.4))  # low > top
        with pytest.raises(ValueError):
            MarketDriftConfig(tier_thresholds=(0.0, 1.5))  # top > 1


# ---------------------------------------------------------------------------
# MarketDriftConfig validation
# ---------------------------------------------------------------------------


class TestMarketDriftConfig:
    def test_defaults_valid(self) -> None:
        cfg = MarketDriftConfig()
        assert cfg.alpha == 0.3
        assert 0.0 < cfg.alpha <= 1.0

    def test_alpha_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(alpha=0.0)

    def test_alpha_above_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(alpha=1.1)

    def test_spillover_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(spillover_adjacent_tier=-0.1)
        with pytest.raises(ValueError):
            MarketDriftConfig(spillover_cross_role=-0.1)

    def test_min_index_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(min_index=0.0)

    def test_max_index_must_be_greater_than_min(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(min_index=1.0, max_index=0.5)

    def test_thresholds_must_have_two_elements(self) -> None:
        with pytest.raises(ValueError):
            MarketDriftConfig(tier_thresholds=(0.4,))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Build initial price index
# ---------------------------------------------------------------------------


class TestBuildInitialPriceIndex:
    def test_all_roles_all_tiers_initialized_to_one(
        self, auction_config: AuctionConfig
    ) -> None:
        idx = build_initial_price_index(auction_config)
        for role in auction_config.role_quotas:
            assert role in idx
            for tier in ALL_TIERS:
                assert idx[role][tier] == 1.0

    def test_initial_index_is_independent_copy(
        self, auction_config: AuctionConfig
    ) -> None:
        idx1 = build_initial_price_index(auction_config)
        idx2 = build_initial_price_index(auction_config)
        idx1["P"]["TOP"] = 0.7
        assert idx2["P"]["TOP"] == 1.0


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_below_min(self) -> None:
        cfg = MarketDriftConfig(min_index=0.5, max_index=1.8)
        assert clamp_index(0.1, cfg) == 0.5

    def test_above_max(self) -> None:
        cfg = MarketDriftConfig(min_index=0.5, max_index=1.8)
        assert clamp_index(2.5, cfg) == 1.8

    def test_within_range(self) -> None:
        cfg = MarketDriftConfig(min_index=0.5, max_index=1.8)
        assert clamp_index(1.0, cfg) == 1.0


# ---------------------------------------------------------------------------
# EWMA update
# ---------------------------------------------------------------------------


class TestUpdatePriceIndex:
    def test_actual_below_expected_lowers_index(self) -> None:
        """TOP pagato meno del previsto -> indice TOP scende."""
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(alpha=0.3, spillover_adjacent_tier=0.0)
        before, after, _ = update_price_index(
            role="P",
            tier="TOP",
            actual_price=10.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        assert before == 1.0
        # new = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        assert after == pytest.approx(0.85)
        assert idx["P"]["TOP"] == pytest.approx(0.85)

    def test_actual_above_expected_raises_index(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(alpha=0.5, spillover_adjacent_tier=0.0)
        _, after, _ = update_price_index(
            role="P",
            tier="TOP",
            actual_price=30.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        # new = 0.5 * 1.0 + 0.5 * 1.5 = 1.25
        assert after == pytest.approx(1.25)

    def test_actual_equal_expected_keeps_index(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(alpha=0.3)
        _, after, _ = update_price_index(
            role="P",
            tier="TOP",
            actual_price=20.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        assert after == pytest.approx(1.0)

    def test_spillover_to_adjacent_tier(self) -> None:
        """TOP scontato deve attenuare anche MID."""
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(
            alpha=0.4, spillover_adjacent_tier=0.25, spillover_cross_role=0.0
        )
        update_price_index(
            role="P",
            tier="TOP",
            actual_price=10.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        # ratio = 0.5
        # adj_new = 1.0 * (1 + 0.25 * (0.5 - 1.0)) = 1.0 * (1 - 0.125) = 0.875
        assert idx["P"]["MID"] == pytest.approx(0.875)

    def test_spillover_cross_role_hook(self) -> None:
        """spillover_cross_role applicato allo stesso tier di altri ruoli."""
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(
            alpha=0.4, spillover_adjacent_tier=0.0, spillover_cross_role=0.2
        )
        update_price_index(
            role="P",
            tier="TOP",
            actual_price=10.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        # P:TOP si aggiorna normalmente
        # D:TOP riceve spillover
        # adj_new = 1.0 * (1 + 0.2 * (0.5 - 1.0)) = 0.9
        assert idx["D"]["TOP"] == pytest.approx(0.9)
        # D:MID e D:LOW non sono toccati (cross-role è solo stesso tier)
        assert idx["D"]["MID"] == 1.0
        assert idx["D"]["LOW"] == 1.0

    def test_spillover_cross_role_disattivato_per_default(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig()  # spillover_cross_role=0.0
        update_price_index(
            role="P",
            tier="TOP",
            actual_price=10.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        assert idx["D"]["TOP"] == 1.0
        assert idx["D"]["MID"] == 1.0
        assert idx["D"]["LOW"] == 1.0

    def test_clamp_lower_bound_after_negative_pressure(self) -> None:
        """Dopo molti step verso il basso, clamp rispetta min_index."""
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(
            alpha=0.5,
            spillover_adjacent_tier=0.5,
            min_index=0.5,
            max_index=1.8,
        )
        for _ in range(5):
            update_price_index(
                role="P",
                tier="TOP",
                actual_price=1.0,
                expected_price=100.0,  # ratio = 0.01
                price_index=idx,
                config=cfg,
            )
        assert idx["P"]["TOP"] >= 0.5

    def test_clamp_upper_bound_after_positive_pressure(self) -> None:
        """Dopo molti step verso l'alto, clamp rispetta max_index."""
        idx = _build_index_for_update()
        cfg = MarketDriftConfig(
            alpha=0.5,
            spillover_adjacent_tier=0.5,
            min_index=0.5,
            max_index=1.8,
        )
        for _ in range(5):
            update_price_index(
                role="P",
                tier="TOP",
                actual_price=1000.0,
                expected_price=1.0,
                price_index=idx,
                config=cfg,
            )
        assert idx["P"]["TOP"] <= 1.8

    def test_snapshot_indipendente_da_mutazioni_successive(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig()
        _, _, snap = update_price_index(
            role="P",
            tier="TOP",
            actual_price=10.0,
            expected_price=20.0,
            price_index=idx,
            config=cfg,
        )
        assert snap["P"]["TOP"] == 1.0
        idx["P"]["TOP"] = 0.1
        assert snap["P"]["TOP"] == 1.0

    def test_actual_negativo_rifiutato(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig()
        with pytest.raises(ValueError):
            update_price_index(
                role="P",
                tier="TOP",
                actual_price=-1.0,
                expected_price=20.0,
                price_index=idx,
                config=cfg,
            )

    def test_expected_non_positivo_rifiutato(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig()
        with pytest.raises(ValueError):
            update_price_index(
                role="P",
                tier="TOP",
                actual_price=10.0,
                expected_price=0.0,
                price_index=idx,
                config=cfg,
            )

    def test_role_sconosciuto_rifiutato(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig()
        with pytest.raises(ValueError):
            update_price_index(
                role="X",  # type: ignore[arg-type]
                tier="TOP",
                actual_price=10.0,
                expected_price=20.0,
                price_index=idx,
                config=cfg,
            )

    def test_tier_sconosciuto_rifiutato(self) -> None:
        idx = _build_index_for_update()
        cfg = MarketDriftConfig()
        with pytest.raises(ValueError):
            update_price_index(
                role="P",
                tier="MEGA",  # type: ignore[arg-type]
                actual_price=10.0,
                expected_price=20.0,
                price_index=idx,
                config=cfg,
            )


# ---------------------------------------------------------------------------
# Baseline / expected price
# ---------------------------------------------------------------------------


class TestComputeBaselineCost:
    def test_senza_inflation_baseline(self) -> None:
        cfg = AuctionConfig(num_participants=4, use_inflation_baseline=False)
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=20,
            projected_score=6.0,
        )
        assert compute_baseline_cost(p, 0.5, cfg) == 20.0

    def test_con_inflation_baseline(self) -> None:
        cfg = AuctionConfig(
            num_participants=8,
            use_inflation_baseline=True,
            inflation_config=InflationConfig(),
        )
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=20,
            projected_score=8.0,
        )
        # Con percentile alto e num_participants=8, l'inflazione statica
        # dell'ottimizzatore deve produrre un costo >= listino nudo.
        baseline = compute_baseline_cost(p, 0.95, cfg)
        assert baseline >= 20.0

    def test_default_300_300_nessuna_scala(self) -> None:
        """Con reference_budget=budget_initial=300 il fattore di scala è 1.0."""
        cfg = AuctionConfig(
            num_participants=4,
            use_inflation_baseline=False,
            reference_budget=300,
            budget_initial=300,
        )
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=20,
            projected_score=6.0,
        )
        assert compute_baseline_cost(p, 0.5, cfg) == 20.0

    def test_scaling_up_quando_budget_maggiore(self) -> None:
        """Listino 300cr in asta 500cr → baseline scalata di 500/300."""
        cfg = AuctionConfig(
            num_participants=4,
            use_inflation_baseline=False,
            reference_budget=300,
            budget_initial=500,
        )
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=30,
            projected_score=6.0,
        )
        assert compute_baseline_cost(p, 0.5, cfg) == pytest.approx(50.0)

    def test_scaling_down_quando_budget_minore(self) -> None:
        """Listino 300cr in asta 100cr → baseline scalata di 100/300."""
        cfg = AuctionConfig(
            num_participants=4,
            use_inflation_baseline=False,
            reference_budget=300,
            budget_initial=100,
        )
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=30,
            projected_score=6.0,
        )
        assert compute_baseline_cost(p, 0.5, cfg) == pytest.approx(10.0)

    def test_scaling_con_inflation_baseline(self) -> None:
        """Lo scaling precede l'inflazione: listino scalato viene inflazionato."""
        cfg = AuctionConfig(
            num_participants=8,
            use_inflation_baseline=True,
            inflation_config=InflationConfig(),
            reference_budget=300,
            budget_initial=600,  # 2× → listino raddoppiato prima dell'inflazione
        )
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=20,
            projected_score=8.0,
        )
        baseline = compute_baseline_cost(p, 0.95, cfg)
        # In asta 600cr il listino è 40; l'inflazione di un top-tier
        # percentile=0.95 non deve scendere sotto il listino scalato.
        assert baseline >= 40.0


class TestAuctionConfigBudgetValidation:
    def test_reference_budget_negativo_rifiutato(self) -> None:
        with pytest.raises(ValueError, match="reference_budget"):
            AuctionConfig(
                num_participants=4,
                reference_budget=-10,
            )

    def test_reference_budget_zero_rifiutato(self) -> None:
        with pytest.raises(ValueError, match="reference_budget"):
            AuctionConfig(
                num_participants=4,
                reference_budget=0,
            )

    def test_budget_initial_negativo_rifiutato(self) -> None:
        with pytest.raises(ValueError, match="budget_initial"):
            AuctionConfig(
                num_participants=4,
                budget_initial=-50,
            )

    def test_budget_initial_zero_rifiutato(self) -> None:
        with pytest.raises(ValueError, match="budget_initial"):
            AuctionConfig(
                num_participants=4,
                budget_initial=0,
            )

    def test_defaults_300_300(self) -> None:
        """I default storici sono entrambi 300 (fattore di scala = 1.0)."""
        cfg = AuctionConfig(num_participants=4)
        assert cfg.reference_budget == 300
        assert cfg.budget_initial == 300


class TestComputeExpectedPrice:
    def test_expected_equals_baseline_con_indice_1(
        self, auction_config: AuctionConfig
    ) -> None:
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=20,
            projected_score=6.0,
        )
        idx = build_initial_price_index(auction_config)
        price = compute_expected_price(p, 0.5, "P", "TOP", idx, auction_config)
        assert price == pytest.approx(20.0)

    def test_expected_cresce_con_indice(
        self, auction_config: AuctionConfig
    ) -> None:
        p = Player(
            player_id="x",
            name="X",
            real_team="T",
            role="P",
            cost=20,
            projected_score=6.0,
        )
        idx = build_initial_price_index(auction_config)
        idx["P"]["TOP"] = 1.5
        price = compute_expected_price(p, 0.5, "P", "TOP", idx, auction_config)
        assert price == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Live projection
# ---------------------------------------------------------------------------


class TestGetCurrentProjection:
    def test_prima_di_assegnazioni(self, goalkeeper_pool: list[Player]) -> None:
        cfg = AuctionConfig(num_participants=4)
        state = initialize_auction(_make_participants(4), cfg, goalkeeper_pool)
        # p_top1 ha projected_score=8.0, percentile top -> TOP tier.
        # Indice 1.0, baseline = cost = 30.
        proj = get_current_projection(state, "p_top1", goalkeeper_pool)
        assert proj == pytest.approx(30.0)

    def test_dopo_assegnazione_sotto_prezzo(
        self, goalkeeper_pool: list[Player]
    ) -> None:
        """TOP GK pagato meno del previsto -> la proiezione sui TOP GK scende."""
        cfg = AuctionConfig(num_participants=4)
        state = initialize_auction(_make_participants(4), cfg, goalkeeper_pool)
        # p_top1 atteso a 30, registrato a 15.
        result = record_assignment(state, "p_top1", "u1", 15)
        assert result.success
        # p_top2 (ancora disponibile, TOP) ora ha proiezione aggiornata.
        proj_top2 = get_current_projection(state, "p_top2", goalkeeper_pool)
        # L'indice P:TOP è sceso (ratio=15/30=0.5, alpha=0.3 -> 0.85).
        # expected per p_top2 = 28 * 0.85 = 23.8
        assert proj_top2 == pytest.approx(28.0 * 0.85)

    def test_unknown_player_solleva(self, goalkeeper_pool: list[Player]) -> None:
        cfg = AuctionConfig(num_participants=4)
        state = initialize_auction(_make_participants(4), cfg, goalkeeper_pool)
        with pytest.raises(ValueError):
            get_current_projection(state, "ghost", goalkeeper_pool)
