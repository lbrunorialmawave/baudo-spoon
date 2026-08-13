from __future__ import annotations

import pytest
import pandas as pd

from ml.data.target import WEIGHTS_BY_ROLE, _BASE_RATING, attach_target, compute_approx_fantavoto


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


# ── min_minutes exemption for cross-league fallback rows ─────────────────────
# (ml/data/loader.py::_append_foreign_fallback_rows — inference-only neo-arrivi
# rows must survive the noisy-target floor that real low-sample Serie A rows
# are still correctly dropped by.)

def _make_season_df(**overrides) -> pd.DataFrame:
    base = {
        "player_fotmob_id": [1],
        "canonical_role": ["FWD"],
        "mins_played": [200],  # well below the 800 min_minutes default
        "goals": [1],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_foreign_fallback_row_survives_min_minutes_floor() -> None:
    df = _make_season_df(is_foreign_fallback=True)
    result = attach_target(df)
    assert len(result) == 1


def test_non_foreign_low_minutes_row_still_dropped() -> None:
    """Legacy path (no hard_floor): rows under the default min_minutes=800
    are dropped. Covers enable_limited_sample_training=False behaviour.
    """
    df = _make_season_df(is_foreign_fallback=False)
    result = attach_target(df)
    assert len(result) == 0


def test_row_without_flag_column_still_dropped() -> None:
    """Backward compatibility: callers that never set is_foreign_fallback
    (the column is absent) must keep the original min_minutes behaviour."""
    df = _make_season_df()
    result = attach_target(df)
    assert len(result) == 0


def test_hard_floor_keeps_limited_cohort() -> None:
    """When hard_floor is supplied (limited-sample path), rows in
    [hard_floor, min_minutes) survive attach_target so they can be
    weighted/shrunk downstream.
    """
    # 300 min is below the legacy 800 floor but above the hard floor of 100.
    df = _make_season_df(mins_played=300, is_foreign_fallback=False)
    result = attach_target(df, min_minutes=800, hard_floor=100)
    assert len(result) == 1
    assert result.iloc[0]["mins_played"] == 300


def test_hard_floor_still_drops_below_floor() -> None:
    """Rows strictly below hard_floor are still dropped even when the
    limited-sample path is active.
    """
    df = _make_season_df(mins_played=50, is_foreign_fallback=False)
    result = attach_target(df, min_minutes=800, hard_floor=100)
    assert len(result) == 0


def test_hard_floor_none_preserves_legacy_drop() -> None:
    """Explicit hard_floor=None is equivalent to the pre-low-sample API."""
    df = _make_season_df(mins_played=300, is_foreign_fallback=False)
    result = attach_target(df, min_minutes=800, hard_floor=None)
    assert len(result) == 0


def test_foreign_fallback_survives_even_with_hard_floor() -> None:
    """Foreign fallback rows must continue to bypass any drop threshold."""
    df = _make_season_df(mins_played=50, is_foreign_fallback=True)
    result = attach_target(df, min_minutes=800, hard_floor=100)
    assert len(result) == 1


# ── WEIGHTS_BY_ROLE completeness ──────────────────────────────────────────────

def test_all_roles_present_in_weights() -> None:
    assert set(WEIGHTS_BY_ROLE.keys()) == {"GK", "DEF", "MID", "FWD"}


def test_all_roles_have_at_least_goal_and_disciplinary_weights() -> None:
    for role, weights in WEIGHTS_BY_ROLE.items():
        assert "goals" in weights, f"{role} missing 'goals' weight"
        assert "yellow_card" in weights, f"{role} missing 'yellow_card' weight"
        assert "red_card" in weights, f"{role} missing 'red_card' weight"
