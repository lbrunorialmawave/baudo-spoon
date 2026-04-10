from __future__ import annotations

import pytest
import pandas as pd

from ml.data.target import WEIGHTS_BY_ROLE, _BASE_RATING, compute_approx_fantavoto


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal player-season DataFrame with given stat values."""
    base = {"appearances": [10], "canonical_role": ["FWD"]}
    base.update({k: [v] for k, v in kwargs.items()})
    return pd.DataFrame(base)


# ── Base rating ───────────────────────────────────────────────────────────────

def test_missing_stats_produce_base_rating() -> None:
    """A row with no stat columns (only appearances) should return the base rating."""
    df = _make_df()
    result = compute_approx_fantavoto(df)
    assert result.iloc[0] == pytest.approx(_BASE_RATING)


# ── Role differentiation ──────────────────────────────────────────────────────

def test_def_goal_bonus_greater_than_fwd() -> None:
    df_def = _make_df(goals=10, canonical_role="DEF")
    df_fwd = _make_df(goals=10, canonical_role="FWD")
    assert compute_approx_fantavoto(df_def).iloc[0] > compute_approx_fantavoto(df_fwd).iloc[0]


def test_mid_goal_bonus_greater_than_fwd() -> None:
    df_mid = _make_df(goals=10, canonical_role="MID")
    df_fwd = _make_df(goals=10, canonical_role="FWD")
    assert compute_approx_fantavoto(df_mid).iloc[0] > compute_approx_fantavoto(df_fwd).iloc[0]


def test_def_goal_bonus_greater_than_mid() -> None:
    df_def = _make_df(goals=10, canonical_role="DEF")
    df_mid = _make_df(goals=10, canonical_role="MID")
    assert compute_approx_fantavoto(df_def).iloc[0] > compute_approx_fantavoto(df_mid).iloc[0]


# ── GK specific ───────────────────────────────────────────────────────────────

def test_gk_clean_sheet_raises_rating() -> None:
    """GK with 10 clean sheets in 10 games (rate = 1.0) and 30 saves."""
    df = _make_df(clean_sheet=10, saves=30, appearances=10, canonical_role="GK")
    rating = compute_approx_fantavoto(df).iloc[0]
    # clean_sheet rate 1.0 × 2.5 + saves 3/match × 0.07 + base 6.0
    assert rating > 7.0


def test_gk_goals_prevented_bonus() -> None:
    df = _make_df(_goals_prevented=10, appearances=10, canonical_role="GK")
    rating = compute_approx_fantavoto(df).iloc[0]
    assert rating > _BASE_RATING


# ── Clip ─────────────────────────────────────────────────────────────────────

def test_rating_clipped_at_upper_10() -> None:
    df = _make_df(goals=100, canonical_role="FWD")
    assert compute_approx_fantavoto(df).iloc[0] == pytest.approx(10.0)


def test_rating_clipped_at_lower_1() -> None:
    df = _make_df(red_card=100, own_goals=100, canonical_role="FWD")
    assert compute_approx_fantavoto(df).iloc[0] == pytest.approx(1.0)


# ── Correct column names (regression against old bugs) ───────────────────────

def test_penalty_scored_uses_correct_column_name() -> None:
    """penalty_scored (not penalties_scored) must produce a non-zero contribution."""
    df_correct = _make_df(penalty_scored=5, canonical_role="FWD")
    df_wrong = _make_df(penalties_scored=5, canonical_role="FWD")
    assert compute_approx_fantavoto(df_correct).iloc[0] > _BASE_RATING
    assert compute_approx_fantavoto(df_wrong).iloc[0] == pytest.approx(_BASE_RATING)


def test_penalty_missed_uses_correct_column_name() -> None:
    df_correct = _make_df(penalty_missed=5, canonical_role="FWD")
    df_wrong = _make_df(penalties_missed=5, canonical_role="FWD")
    assert compute_approx_fantavoto(df_correct).iloc[0] < _BASE_RATING
    assert compute_approx_fantavoto(df_wrong).iloc[0] == pytest.approx(_BASE_RATING)


def test_clean_sheet_uses_correct_column_name() -> None:
    df_correct = _make_df(clean_sheet=10, appearances=10, canonical_role="GK")
    df_wrong = _make_df(clean_sheets=10, appearances=10, canonical_role="GK")
    assert compute_approx_fantavoto(df_correct).iloc[0] > _BASE_RATING
    assert compute_approx_fantavoto(df_wrong).iloc[0] == pytest.approx(_BASE_RATING)


# ── Missing canonical_role falls back to FWD ─────────────────────────────────

def test_missing_role_column_falls_back_to_fwd(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    df = pd.DataFrame({"appearances": [10], "goals": [5]})
    with caplog.at_level(logging.WARNING, logger="ml.data.target"):
        result = compute_approx_fantavoto(df)
    expected = compute_approx_fantavoto(
        pd.DataFrame({"appearances": [10], "goals": [5], "canonical_role": ["FWD"]})
    )
    assert result.iloc[0] == pytest.approx(expected.iloc[0])
    assert any("canonical_role" in r.message for r in caplog.records)


# ── WEIGHTS_BY_ROLE completeness ──────────────────────────────────────────────

def test_all_roles_present_in_weights() -> None:
    assert set(WEIGHTS_BY_ROLE.keys()) == {"GK", "DEF", "MID", "FWD"}


def test_all_roles_have_at_least_goal_and_disciplinary_weights() -> None:
    for role, weights in WEIGHTS_BY_ROLE.items():
        assert "goals" in weights, f"{role} missing 'goals' weight"
        assert "yellow_card" in weights, f"{role} missing 'yellow_card' weight"
        assert "red_card" in weights, f"{role} missing 'red_card' weight"
