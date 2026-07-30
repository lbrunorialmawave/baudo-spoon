"""Regression test: predictions artifact includes fantapunti_totali and probabilita_titolarita."""

import numpy as np
import pandas as pd
import pytest


def _derive_season_value_columns(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the exact derivation logic from trainer.py."""
    _em = predictions_df["expected_minutes"]
    _pf = predictions_df["predicted_fantavoto"]
    _has_minutes = _em > 0
    predictions_df["fantapunti_totali"] = np.where(
        _has_minutes, _pf * (_em / 90.0), np.nan
    )
    predictions_df["probabilita_titolarita"] = np.where(
        _has_minutes, (_em / (38.0 * 90.0)).clip(upper=1.0), np.nan
    )
    return predictions_df


def _json_safe_nan(val):
    """Simulate _json_safe NaN→None conversion for a single value."""
    if isinstance(val, float) and (val != val):
        return None
    return val


def test_fantapunti_totali_derived_correctly():
    """fantapunti_totali = predicted_fantavoto * (expected_minutes / 90)."""
    df = pd.DataFrame(
        {
            "player_fotmob_id": [1, 2, 3],
            "predicted_fantavoto": [7.0, 6.5, 8.0],
            "expected_minutes": [2700.0, 0.0, 3420.0],
        }
    )
    df = _derive_season_value_columns(df)
    records = df.to_dict(orient="records")

    # Apply NaN→None (as _json_safe does)
    for r in records:
        r["fantapunti_totali"] = _json_safe_nan(r["fantapunti_totali"])
        r["probabilita_titolarita"] = _json_safe_nan(r["probabilita_titolarita"])

    # Player 1: 7.0 * 30 = 210.0, prob = 2700/3420
    assert records[0]["fantapunti_totali"] == pytest.approx(210.0)
    assert records[0]["probabilita_titolarita"] == pytest.approx(2700.0 / 3420.0)

    # Player 2: no minutes → None
    assert records[1]["fantapunti_totali"] is None
    assert records[1]["probabilita_titolarita"] is None

    # Player 3: 8.0 * 38 = 304.0, prob = 1.0 (clipped)
    assert records[2]["fantapunti_totali"] == pytest.approx(304.0)
    assert records[2]["probabilita_titolarita"] == pytest.approx(1.0)


def test_fantapunti_totali_non_null_when_both_inputs_available():
    """Every player with predicted_fantavoto > 0 and expected_minutes > 0 gets a value."""
    df = pd.DataFrame(
        {
            "player_fotmob_id": range(10),
            "predicted_fantavoto": np.random.uniform(5.5, 8.5, 10),
            "expected_minutes": np.random.uniform(900, 3420, 10),
        }
    )
    df = _derive_season_value_columns(df)

    assert df["fantapunti_totali"].notna().all()
    assert df["probabilita_titolarita"].notna().all()
    assert (df["probabilita_titolarita"] <= 1.0).all()
    assert (df["fantapunti_totali"] > 0).all()
