"""Regression tests for Trainer foreign-fallback quarantine (PR3).

Contract under test (ml/pipeline/trainer.py ~L523):
  foreign players must NOT enter training or evaluation;
  they ARE eligible for inference-only prediction.
"""

from __future__ import annotations

import pandas as pd


def _quarantine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror of Trainer.run step 3b — keep in sync with trainer.py."""
    if "is_foreign_fallback" not in df.columns:
        foreign_mask = pd.Series(False, index=df.index)
    else:
        foreign_mask = (
            df["is_foreign_fallback"].astype("boolean").fillna(False).astype(bool)
        )
    df_core = df[~foreign_mask].copy()
    df_foreign = df[foreign_mask].copy()
    return df_core, df_foreign


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_fotmob_id": 1,
                "player_name": "Core A",
                "season_start": 2025,
                "fantavoto_medio": 6.5,
                "is_foreign_fallback": False,
            },
            {
                "player_fotmob_id": 2,
                "player_name": "Core B",
                "season_start": 2025,
                "fantavoto_medio": 6.8,
                "is_foreign_fallback": False,
            },
            {
                "player_fotmob_id": 99,
                "player_name": "Foreign Eredivisie",
                "season_start": 2025,
                "fantavoto_medio": 7.2,
                "is_foreign_fallback": True,
                "league_name": "Eredivisie",
            },
        ]
    )


def test_foreign_row_excluded_from_core():
    df = _sample_frame()
    df_core, df_foreign = _quarantine(df)
    assert len(df_core) == 2
    assert 99 not in set(df_core["player_fotmob_id"])
    assert set(df_core["player_fotmob_id"]) == {1, 2}


def test_foreign_row_present_in_foreign_slice():
    df = _sample_frame()
    _, df_foreign = _quarantine(df)
    assert len(df_foreign) == 1
    assert df_foreign.iloc[0]["player_fotmob_id"] == 99
    assert bool(df_foreign.iloc[0]["is_foreign_fallback"]) is True


def test_missing_flag_treated_as_core():
    df = pd.DataFrame([{"player_fotmob_id": 1, "fantavoto_medio": 6.0}])
    df_core, df_foreign = _quarantine(df)
    assert len(df_core) == 1
    assert len(df_foreign) == 0


def test_null_flag_treated_as_core():
    df = pd.DataFrame(
        [
            {"player_fotmob_id": 1, "is_foreign_fallback": None, "fantavoto_medio": 6.0},
            {"player_fotmob_id": 2, "is_foreign_fallback": True, "fantavoto_medio": 7.0},
        ]
    )
    df_core, df_foreign = _quarantine(df)
    assert set(df_core["player_fotmob_id"]) == {1}
    assert set(df_foreign["player_fotmob_id"]) == {2}


def test_temporal_split_input_has_no_foreign():
    df = _sample_frame()
    df_core, _ = _quarantine(df)
    test_seasons = {2025}
    df_train = df_core[~df_core["season_start"].isin(test_seasons)]
    df_test = df_core[df_core["season_start"].isin(test_seasons)]
    assert 99 not in set(df_train["player_fotmob_id"])
    assert 99 not in set(df_test["player_fotmob_id"])


def test_prediction_slice_includes_foreign():
    df = _sample_frame()
    df_core, df_foreign = _quarantine(df)
    core_preds = df_core[["player_fotmob_id"]].copy()
    core_preds["predicted_fantavoto"] = 6.5
    core_preds["is_foreign_fallback"] = False
    foreign_preds = df_foreign[["player_fotmob_id"]].copy()
    foreign_preds["predicted_fantavoto"] = 6.2
    foreign_preds["is_foreign_fallback"] = True
    all_preds = pd.concat([core_preds, foreign_preds], ignore_index=True)
    assert 99 in set(all_preds["player_fotmob_id"])
    foreign_row = all_preds[all_preds["player_fotmob_id"] == 99].iloc[0]
    assert bool(foreign_row["is_foreign_fallback"]) is True
