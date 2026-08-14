from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.sample_reliability.cohort import (
    COHORT_INSUFFICIENT,
    COHORT_LIMITED,
    COHORT_STANDARD,
)
from ml.sample_reliability.output_reliability import attach_output_reliability


def _make_df() -> pd.DataFrame:
    """20 STANDARD rows (predicted tightly around 6.0) + one LIMITED
    outlier (300 min, predicted 9.5 — a "phenom" reading from a small
    sample) + one INSUFFICIENT row (40 min).
    """
    rng = np.random.default_rng(42)
    n_standard = 20
    standard_rows = {
        "player_fotmob_id": list(range(n_standard)),
        "player_name": [f"Standard {i}" for i in range(n_standard)],
        "canonical_role": ["MID"] * n_standard,
        "mins_played": [1000 + i * 10 for i in range(n_standard)],
        "predicted_fantavoto": list(6.0 + rng.normal(0, 0.1, n_standard)),
    }
    df = pd.DataFrame(standard_rows)
    outlier = pd.DataFrame({
        "player_fotmob_id": [999],
        "player_name": ["Hot Streak"],
        "canonical_role": ["MID"],
        "mins_played": [300],
        "predicted_fantavoto": [9.5],
    })
    insufficient = pd.DataFrame({
        "player_fotmob_id": [998],
        "player_name": ["Barely Played"],
        "canonical_role": ["MID"],
        "mins_played": [40],
        "predicted_fantavoto": [7.0],
    })
    return pd.concat([df, outlier, insufficient], ignore_index=True)


def test_cohort_labels_assigned_correctly() -> None:
    df = _make_df()
    out, meta = attach_output_reliability(df, predicted_col="predicted_fantavoto")

    standard_rows = out[out["player_fotmob_id"] < 900]
    assert (standard_rows["sample_cohort"] == COHORT_STANDARD).all()
    assert not standard_rows["ml_values_noisy"].any()

    outlier_row = out[out["player_fotmob_id"] == 999].iloc[0]
    assert outlier_row["sample_cohort"] == COHORT_LIMITED
    assert outlier_row["ml_values_noisy"] is np.True_ or outlier_row["ml_values_noisy"] is True

    insufficient_row = out[out["player_fotmob_id"] == 998].iloc[0]
    assert insufficient_row["sample_cohort"] == COHORT_INSUFFICIENT
    assert bool(insufficient_row["ml_values_noisy"]) is True


def test_display_value_damped_toward_median_for_limited_row() -> None:
    df = _make_df()
    out, _ = attach_output_reliability(df, predicted_col="predicted_fantavoto")

    outlier_row = out[out["player_fotmob_id"] == 999].iloc[0]
    # Raw prediction (9.5) should be pulled well below its face value,
    # toward the ~6.0 STANDARD-cohort median — but not collapsed to
    # exactly the median either (there IS a 300-minute signal).
    assert outlier_row["predicted_fantavoto_display"] < 9.5
    assert outlier_row["predicted_fantavoto_display"] > 6.0


def test_standard_rows_display_value_near_unchanged() -> None:
    df = _make_df()
    out, _ = attach_output_reliability(df, predicted_col="predicted_fantavoto")

    standard_rows = out[out["player_fotmob_id"] < 900]
    # minutes (>=1000) >> prior_strength (default 300) => shrinkage
    # pulls the display value only slightly toward the prior.
    diff = (standard_rows["predicted_fantavoto_display"] - standard_rows["predicted_fantavoto"]).abs()
    assert (diff < 0.15).all()


def test_raw_predicted_column_untouched() -> None:
    df = _make_df()
    out, _ = attach_output_reliability(df, predicted_col="predicted_fantavoto")
    pd.testing.assert_series_equal(
        out["predicted_fantavoto"], df["predicted_fantavoto"], check_names=True,
    )


def test_missing_columns_is_safe_noop() -> None:
    df = pd.DataFrame({"player_name": ["A"], "predicted_fantavoto": [7.0]})
    out, meta = attach_output_reliability(df, predicted_col="predicted_fantavoto")
    assert meta["enabled"] is False
    assert (out["sample_cohort"] == COHORT_STANDARD).all()
    assert not out["ml_values_noisy"].any()
    assert out["predicted_fantavoto_display"].iloc[0] == 7.0


def test_role_group_falls_back_to_global_when_too_few_standard_rows() -> None:
    """A GK role with only 2 STANDARD rows must borrow the global prior,
    not compute a degenerate median-of-2."""
    df = _make_df()
    gk = pd.DataFrame({
        "player_fotmob_id": [1001, 1002, 1003],
        "player_name": ["GK1", "GK2", "GK-Limited"],
        "canonical_role": ["GK", "GK", "GK"],
        "mins_played": [1200, 1300, 250],
        "predicted_fantavoto": [6.1, 6.3, 9.0],
    })
    df = pd.concat([df, gk], ignore_index=True)
    out, meta = attach_output_reliability(
        df, predicted_col="predicted_fantavoto", role_col="canonical_role",
        min_standard_rows_for_prior=30,
    )
    gk_limited = out[out["player_fotmob_id"] == 1003].iloc[0]
    # Prior for GK fell back to the global (MID-dominated, ~6.0) median
    # since GK only has 2 STANDARD rows (< 30) — not the GK-only mean.
    assert meta["priors_by_role"]["GK"] == pytest.approx(meta["priors_by_role"]["MID"], abs=1e-6)
    assert gk_limited["predicted_fantavoto_display"] < 9.0


def test_exclude_from_prior_mask_does_not_skip_classification() -> None:
    df = _make_df()
    exclude = pd.Series(False, index=df.index)
    exclude.loc[df["player_fotmob_id"] < 900] = True  # exclude all STANDARD rows
    out, meta = attach_output_reliability(
        df, predicted_col="predicted_fantavoto", exclude_from_prior_mask=exclude,
    )
    # Excluded rows are still classified/damped, just don't feed the prior.
    assert len(out) == len(df)
    assert meta["priors_by_role"]  # prior still computed (falls back globally)
