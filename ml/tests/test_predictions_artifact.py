"""Regression tests for the season-value derivation in the predictions artefact.

The two helpers under test live in :mod:`ml.domain.predictions` and are
re-used by:

* the training pipeline (``ml/pipeline/trainer.py``) which writes the
  columns into ``results_latest.json``;
* the MANTRA runner (``ml/mantra/runner.py``) which projects the same
  numbers onto the MANTRA artefact;
* the API's optimizer pool (``api/src/data_repository.py``) which reads
  them back to expose ``season_value`` / ``start_probability``.

These tests guard the contract for both the vectorised writer
(``derive_season_value_columns``) and the per-record consumer
(``resolve_season_value_fields``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.domain.predictions import (
    derive_season_value_columns,
    resolve_season_value_fields,
)


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
    df = derive_season_value_columns(df)
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
    df = derive_season_value_columns(df)

    assert df["fantapunti_totali"].notna().all()
    assert df["probabilita_titolarita"].notna().all()
    assert (df["probabilita_titolarita"] <= 1.0).all()
    assert (df["fantapunti_totali"] > 0).all()


# ── resolve_season_value_fields: single-record consumer side ───────────────


def test_resolve_season_value_fields_none_when_prediction_is_none():
    """``None`` input short-circuits to ``(None, None)``."""
    assert resolve_season_value_fields(None) == (None, None)


def test_resolve_season_value_fields_prefers_artifact_columns():
    """Pre-computed ``fantapunti_totali`` / ``probabilita_titolarita`` win."""
    pred = {
        "player_fotmob_id": 1,
        "predicted_fantavoto": 7.0,
        "expected_minutes": 2700.0,
        "fantapunti_totali": 200.0,
        "probabilita_titolarita": 0.85,
    }
    sv, sp = resolve_season_value_fields(pred)
    assert sv == pytest.approx(200.0)
    assert sp == pytest.approx(0.85)


def test_resolve_season_value_fields_derives_when_artifact_missing():
    """When the artifact has no pre-computed values, derive them."""
    pred = {
        "player_fotmob_id": 1,
        "predicted_fantavoto": 7.0,
        "expected_minutes": 2700.0,
    }
    sv, sp = resolve_season_value_fields(pred)
    assert sv == pytest.approx(210.0)  # 7.0 * 30
    assert sp == pytest.approx(2700.0 / 3420.0)


def test_resolve_season_value_fields_none_when_inputs_missing():
    """No minutes, no pre-computed value → both None."""
    pred = {"player_fotmob_id": 1, "predicted_fantavoto": 7.0, "expected_minutes": 0.0}
    sv, sp = resolve_season_value_fields(pred)
    assert sv is None
    assert sp == pytest.approx(0.0)  # em=0, ≥ 0 → 0/3420 = 0.0


def test_resolve_season_value_fields_falls_back_to_caller_score():
    """When the prediction lacks ``predicted_fantavoto``, the caller's
    ``fallback_predicted_score`` is used for the derivation, matching
    the historical ``data_repository.get_player_pool`` behaviour."""
    pred = {"player_fotmob_id": 1, "expected_minutes": 2700.0}  # no predicted_fantavoto
    sv, _ = resolve_season_value_fields(pred, fallback_predicted_score=6.5)
    assert sv == pytest.approx(6.5 * 30.0)


def test_resolve_season_value_fields_treats_nan_as_missing():
    """NaN in the artifact should be ignored, not propagated."""
    pred = {
        "player_fotmob_id": 1,
        "predicted_fantavoto": 7.0,
        "expected_minutes": float("nan"),
        "fantapunti_totali": float("nan"),
        "probabilita_titolarita": float("nan"),
    }
    sv, sp = resolve_season_value_fields(pred)
    assert sv is None
    assert sp is None
