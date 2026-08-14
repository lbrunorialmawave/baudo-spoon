"""Regression tests for the Trainer Step 12 output-reliability wiring.

Contract under test (ml/pipeline/trainer.py ~L1375 and ~L1292):
  predictions_df and next_season_predictions must both carry
  sample_cohort / ml_values_noisy / <predicted_col>_display columns,
  the raw predicted_fantavoto / predicted_next_fantavoto columns must
  be untouched, and foreign-fallback rows must be classified/damped
  but excluded from prior estimation.

Mirrors the call the trainer actually makes (same kwargs) rather than
re-deriving the logic, so a signature drift in
``attach_output_reliability`` fails this test instead of silently
breaking the pipeline — same style as
``test_trainer_foreign_quarantine.py``.
"""

from __future__ import annotations

import pandas as pd

from ml.sample_reliability.output_reliability import attach_output_reliability


def _predictions_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "player_fotmob_id": 1, "player_name": "Standard Regular",
            "canonical_role": "MID", "fantavoto_medio": 6.2,
            "predicted_fantavoto": 6.1, "expected_minutes": 2400,
            "is_foreign_fallback": False,
        },
        {
            "player_fotmob_id": 2, "player_name": "Standard Regular 2",
            "canonical_role": "MID", "fantavoto_medio": 6.4,
            "predicted_fantavoto": 6.3, "expected_minutes": 2100,
            "is_foreign_fallback": False,
        },
        {
            "player_fotmob_id": 3, "player_name": "Small Sample Hot Streak",
            "canonical_role": "MID", "fantavoto_medio": 7.0,
            "predicted_fantavoto": 9.2, "expected_minutes": 180,
            "is_foreign_fallback": False,
        },
        {
            "player_fotmob_id": 99, "player_name": "Neo-arrivo Foreign",
            "canonical_role": "FWD", "fantavoto_medio": 6.8,
            "predicted_fantavoto": 6.8, "expected_minutes": 0,
            "is_foreign_fallback": True,
        },
    ])


def test_predictions_df_wiring_adds_expected_columns() -> None:
    df = _predictions_frame()
    out, meta = attach_output_reliability(
        df,
        predicted_col="predicted_fantavoto",
        minutes_col="expected_minutes",
        role_col="canonical_role",
        min_minutes_hard=100,
        standard_minutes=800,
        prior_strength=300,
        exclude_from_prior_mask=df["is_foreign_fallback"],
    )

    assert {"sample_cohort", "ml_values_noisy", "predicted_fantavoto_display"} <= set(out.columns)

    # Raw column untouched.
    pd.testing.assert_series_equal(
        out["predicted_fantavoto"], df["predicted_fantavoto"], check_names=True,
    )

    hot_streak = out[out["player_fotmob_id"] == 3].iloc[0]
    assert bool(hot_streak["ml_values_noisy"]) is True
    assert hot_streak["predicted_fantavoto_display"] < hot_streak["predicted_fantavoto"]

    standard = out[out["player_fotmob_id"] == 1].iloc[0]
    assert bool(standard["ml_values_noisy"]) is False

    # Foreign row (0 minutes) is INSUFFICIENT: classified/damped, but
    # excluded from prior estimation via exclude_from_prior_mask.
    foreign = out[out["player_fotmob_id"] == 99].iloc[0]
    assert bool(foreign["ml_values_noisy"]) is True


def test_next_season_predictions_wiring_adds_expected_columns() -> None:
    """Same contract, mirrors the ml.pipeline.trainer 'predict next
    season' call which uses predicted_next_fantavoto / mins_played."""
    df_next = pd.DataFrame([
        {
            "player_fotmob_id": 1, "player_name": "Regular",
            "canonical_role": "DEF", "mins_played": 2600,
            "predicted_next_fantavoto": 6.0,
            "is_foreign_fallback": False,
        },
        {
            "player_fotmob_id": 2, "player_name": "Breakout Candidate",
            "canonical_role": "DEF", "mins_played": 220,
            "predicted_next_fantavoto": 8.8,
            "is_foreign_fallback": False,
        },
    ])
    out, meta = attach_output_reliability(
        df_next,
        predicted_col="predicted_next_fantavoto",
        minutes_col="mins_played",
        role_col="canonical_role",
        min_minutes_hard=100,
        standard_minutes=800,
        prior_strength=300,
        exclude_from_prior_mask=df_next["is_foreign_fallback"],
    )
    assert "predicted_next_fantavoto_display" in out.columns
    candidate = out[out["player_fotmob_id"] == 2].iloc[0]
    assert candidate["sample_cohort"] == "LIMITED"
    assert candidate["predicted_next_fantavoto_display"] < 8.8
