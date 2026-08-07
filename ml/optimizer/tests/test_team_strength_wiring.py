"""Tests proving team_strength_scores reaches solver, price_drift, win_probability.

Process guard: these are entry-point reachability tests — they prove the Elo
signal actually affects computation outputs, not just that the functions accept
the parameter.
"""

from __future__ import annotations

from ml.optimizer.models import (
    Formation,
    InflationConfig,
    OptimizationConfig,
    Player,
    StrategyProfile,
)
from ml.optimizer.solver import solve_strategy
from ml.optimizer.team_strength import load_team_strength_scores
from ml.optimizer.win_probability import (
    WinProbabilityConfig,
    estimate_completion_probability,
)


def _make_player(pid: str, team: str, cost: int = 15, role: str = "A") -> Player:
    return Player(
        player_id=pid,
        name=pid,
        role=role,
        real_team=team,
        cost=cost,
        projected_score=7.5,
    )


_ELO_CONFIG = InflationConfig(
    inflation_percentile_threshold=0.5,
    max_inflation_multiplier=2.0,
    base_inflation_rate=0.05,
    baseline_participants=8,
    team_strength_multiplier=0.3,
)


def _small_pool() -> list[Player]:
    """25-player pool with mix of Inter (high Elo) and Lecce (low Elo)."""
    pool: list[Player] = []
    teams = ["Inter", "Lecce", "Milan", "Napoli", "Roma", "Juventus", "Fiorentina"]
    role_counts = {"P": 3, "D": 8, "C": 8, "A": 6}
    idx = 0
    for role, count in role_counts.items():
        for i in range(count):
            team = teams[idx % len(teams)]
            idx += 1
            pool.append(
                Player(
                    player_id=f"{role}{i}",
                    name=f"{role}{i}",
                    role=role,
                    real_team=team,
                    cost=10,
                    projected_score=6.5 + i * 0.1,
                )
            )
    return pool


class TestSolverUsesTeamStrength:
    """Task 1: solver's internal effective_cost dict reflects Elo adjustment."""

    def test_effective_cost_differs_with_elo(self) -> None:
        pool = _small_pool()
        config_with_elo = OptimizationConfig(
            budget=500,
            formations=[Formation("3-4-3", 3, 4, 3), Formation("4-3-3", 4, 3, 3)],
            num_participants=8,
            max_players_per_team=6,
            big_teams=frozenset(),
            big_teams_cap=25,
            min_distinct_teams=2,
            inflation_config=_ELO_CONFIG,
        )
        config_no_elo = OptimizationConfig(
            budget=500,
            formations=[Formation("3-4-3", 3, 4, 3), Formation("4-3-3", 4, 3, 3)],
            num_participants=8,
            max_players_per_team=6,
            big_teams=frozenset(),
            big_teams_cap=25,
            min_distinct_teams=2,
            inflation_config=InflationConfig(
                inflation_percentile_threshold=0.5,
                max_inflation_multiplier=2.0,
                base_inflation_rate=0.05,
                baseline_participants=8,
                team_strength_multiplier=0.0,
            ),
        )
        strategy = StrategyProfile(
            name="balanced", role_weight={"P": 1, "D": 1, "C": 1, "A": 1}
        )

        result_elo = solve_strategy(pool, config_with_elo, strategy)
        result_no_elo = solve_strategy(pool, config_no_elo, strategy)

        # With Elo active, effective costs change → different total_effective_cost
        assert result_elo.total_effective_cost != result_no_elo.total_effective_cost


class TestPriceDriftUsesTeamStrength:
    """Task 1: compute_baseline_cost reflects Elo when team_strength_scores passed."""

    def test_baseline_cost_with_elo(self) -> None:
        from ml.auction.models import AuctionConfig
        from ml.auction.price_drift import compute_baseline_cost

        cfg = AuctionConfig(
            num_participants=8,
            use_inflation_baseline=True,
            inflation_config=_ELO_CONFIG,
        )
        p_inter = _make_player("x", "Inter")
        p_lecce = _make_player("y", "Lecce")
        ts = load_team_strength_scores(known_teams={"Inter", "Lecce"})

        cost_inter = compute_baseline_cost(p_inter, 0.9, cfg, team_strength_scores=ts)
        cost_lecce = compute_baseline_cost(p_lecce, 0.9, cfg, team_strength_scores=ts)

        assert cost_inter > cost_lecce


class TestWinProbabilityUsesTeamStrength:
    """Task 1: estimate_completion_probability uses Elo for effective cost."""

    def test_probability_differs_with_elo(self) -> None:
        import random

        # Need mix of teams so normalization produces non-zero scores
        squad = [
            _make_player("a", "Inter", cost=20),
            _make_player("b", "Lecce", cost=20),
            _make_player("c", "Inter", cost=20),
        ]
        # Give them different scores so percentiles spread above threshold
        squad[0] = Player("a", "A1", "A", "Inter", 20, 9.0)
        squad[1] = Player("b", "B1", "A", "Lecce", 20, 7.0)
        squad[2] = Player("c", "C1", "A", "Inter", 20, 8.0)

        wp_config = WinProbabilityConfig(n_simulations=5000)
        ts = load_team_strength_scores(known_teams={"Inter", "Lecce"})

        random.seed(42)
        prob_with_elo = estimate_completion_probability(
            squad,
            budget=65,
            config=wp_config,
            inflation_config=_ELO_CONFIG,
            num_participants=10,
            team_strength_scores=ts,
        )
        random.seed(42)
        prob_no_elo = estimate_completion_probability(
            squad,
            budget=65,
            config=wp_config,
            inflation_config=InflationConfig(
                inflation_percentile_threshold=0.5,
                max_inflation_multiplier=2.0,
                base_inflation_rate=0.05,
                baseline_participants=8,
                team_strength_multiplier=0.0,
            ),
            num_participants=10,
            team_strength_scores=None,
        )

        # With Elo, Inter players cost more → lower completion probability
        assert prob_with_elo < prob_no_elo
