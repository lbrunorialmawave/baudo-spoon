"""Tests for team strength score loading and normalization."""

from __future__ import annotations

from pathlib import Path

from ml.optimizer.team_strength import load_team_strength_scores


def test_load_normalizes_to_0_1() -> None:
    """Scores are min-max normalized: lowest club = 0.0, highest = 1.0."""
    scores = load_team_strength_scores()
    assert scores, "Should load at least some clubs"
    assert min(scores.values()) == 0.0
    assert max(scores.values()) == 1.0


def test_inter_is_highest() -> None:
    """Inter has highest Elo in the dataset (1889), so normalized = 1.0."""
    scores = load_team_strength_scores()
    assert scores["Inter"] == 1.0


def test_filter_known_teams() -> None:
    """Only clubs in known_teams are returned."""
    scores = load_team_strength_scores(known_teams={"Inter", "Milan", "FakeClub"})
    assert "Inter" in scores
    assert "Milan" in scores
    assert "FakeClub" not in scores
    assert "Napoli" not in scores


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    """Non-existent path returns empty dict."""
    scores = load_team_strength_scores(path=tmp_path / "nope.json")
    assert scores == {}


def test_inflation_uses_team_strength() -> None:
    """Two identical players from different clubs get different effective cost when multiplier > 0."""
    from ml.optimizer.inflation import estimate_effective_cost
    from ml.optimizer.models import InflationConfig, Player

    cfg = InflationConfig(
        inflation_percentile_threshold=0.5,
        max_inflation_multiplier=2.0,
        base_inflation_rate=0.05,
        baseline_participants=8,
        team_strength_multiplier=0.3,
    )
    p_inter = Player(player_id="x", name="X", role="A", real_team="Inter", cost=20, projected_score=8.0)
    p_lecce = Player(player_id="y", name="Y", role="A", real_team="Lecce", cost=20, projected_score=8.0)

    ts = load_team_strength_scores(known_teams={"Inter", "Lecce"})

    cost_inter = estimate_effective_cost(p_inter, 0.9, 10, cfg, team_strength_scores=ts)
    cost_lecce = estimate_effective_cost(p_lecce, 0.9, 10, cfg, team_strength_scores=ts)

    assert cost_inter > cost_lecce, "Inter player should cost more due to team strength"


def test_zero_multiplier_no_effect() -> None:
    """When team_strength_multiplier=0.0, team strength has no effect (backward compat)."""
    from ml.optimizer.inflation import estimate_effective_cost
    from ml.optimizer.models import InflationConfig, Player

    cfg = InflationConfig(team_strength_multiplier=0.0)
    p = Player(player_id="x", name="X", role="A", real_team="Inter", cost=20, projected_score=8.0)

    ts = load_team_strength_scores()
    cost_with = estimate_effective_cost(p, 0.9, 10, cfg, team_strength_scores=ts)
    cost_without = estimate_effective_cost(p, 0.9, 10, cfg, team_strength_scores=None)

    assert cost_with == cost_without
