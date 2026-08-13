"""Unit tests for the trainer-side wiring of the low-sample feature flags.

Covers the three flags that were previously read but had no effect on
production training (see ``plan.md`` PR3/PR4/PR7):

* ``enable_shrinkage``       — :meth:`Trainer._apply_shrinkage`
* ``enable_recent_role_features`` — role/opportunity features reaching
  ``select_features`` via ``extra_numeric_candidates``
* ``enable_breakout_model``  — :meth:`Trainer._run_breakout_model`

Also covers the attach_target hard_floor wiring (PR wiring fix) so that
the LIMITED cohort (100–799 min) actually reaches weighting/shrinkage
when ``enable_limited_sample_training`` is True.

These tests exercise the Trainer helper methods and the attach_target
call-site contract directly (no DB, no full ``Trainer.run()``) to stay
fast and hermetic, mirroring the style of
``test_trainer_foreign_quarantine.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.config import MLConfig
from ml.data.target import attach_target
from ml.pipeline.trainer import Trainer
from ml.preprocessing.features import select_features
from ml.preprocessing.role_features import (
    RoleOpportunityFeatureTransformer,
    add_role_opportunity_features,
)


def _cfg(tmp_path: Path, **overrides) -> MLConfig:
    return MLConfig(
        database_url="postgresql://user:pass@localhost/db",
        artifacts_dir=tmp_path,
        test_seasons=1,
        **overrides,
    )


def _toy_frame(n_players: int = 30, seasons: tuple[int, ...] = (2021, 2022, 2023, 2024)) -> pd.DataFrame:
    """Deterministic multi-season, multi-role player frame with per-90 stats."""
    rng = np.random.default_rng(42)
    rows = []
    for p in range(1, n_players + 1):
        role = "GK" if p % 10 == 0 else ("DEF" if p % 3 == 0 else "MID")
        for season in seasons:
            mins = int(rng.integers(50, 3000))
            rows.append({
                "player_fotmob_id": p,
                "player_name": f"Player {p}",
                "team_name": "Team",
                "season_start": season,
                "canonical_role": role,
                "mins_played": mins,
                "starts": max(0, mins // 90 - 1),
                "appearances": max(1, mins // 70),
                "is_foreign_fallback": False,
                "goals_per90": float(rng.uniform(0, 3)),
                "goal_assist_per90": float(rng.uniform(0, 2)),
                "fantavoto_medio": float(rng.uniform(5, 8)),
            })
    return pd.DataFrame(rows)


# ── enable_shrinkage ─────────────────────────────────────────────────────────

class TestApplyShrinkage:
    def test_noop_when_flag_disabled(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, enable_limited_sample_training=True, enable_shrinkage=False)
        trainer = Trainer(cfg)
        df = _toy_frame()
        before = df["goals_per90"].copy()
        meta = trainer._apply_shrinkage(df, prior_exclude_mask=df["is_foreign_fallback"])
        assert meta == {"enabled": False}
        pd.testing.assert_series_equal(df["goals_per90"], before)

    def test_noop_when_limited_sample_training_disabled(self, tmp_path: Path) -> None:
        # enable_shrinkage=True alone must not activate shrinkage — mirrors
        # the documented no-op contract on MLConfig.enable_shrinkage.
        cfg = _cfg(tmp_path, enable_limited_sample_training=False, enable_shrinkage=True)
        trainer = Trainer(cfg)
        df = _toy_frame()
        before = df["goals_per90"].copy()
        meta = trainer._apply_shrinkage(df, prior_exclude_mask=df["is_foreign_fallback"])
        assert meta["enabled"] is False
        pd.testing.assert_series_equal(df["goals_per90"], before)

    def test_enabled_adjusts_per90_columns(self, tmp_path: Path) -> None:
        cfg = _cfg(
            tmp_path,
            enable_limited_sample_training=True,
            enable_shrinkage=True,
            min_minutes=800,
            min_minutes_hard=100,
            shrinkage_prior_strength=300,
        )
        trainer = Trainer(cfg)
        df = _toy_frame()
        before = df["goals_per90"].copy()
        meta = trainer._apply_shrinkage(df, prior_exclude_mask=df["is_foreign_fallback"])
        assert meta["enabled"] is True
        assert "goals_per90" in meta["columns"]
        assert not df["goals_per90"].equals(before)

    def test_extreme_low_minutes_pulled_toward_prior(self, tmp_path: Path) -> None:
        """A 50-minute outlier rate should move toward the role's prior."""
        cfg = _cfg(
            tmp_path,
            enable_limited_sample_training=True,
            enable_shrinkage=True,
            min_minutes=800,
            shrinkage_prior_strength=300,
        )
        trainer = Trainer(cfg)
        df = _toy_frame(n_players=40)
        # Inject a single extreme low-sample outlier.
        outlier_idx = df.index[0]
        df.loc[outlier_idx, "mins_played"] = 50
        df.loc[outlier_idx, "goals_per90"] = 9.0  # absurd small-sample rate
        raw_value = 9.0
        trainer._apply_shrinkage(df, prior_exclude_mask=df["is_foreign_fallback"])
        adjusted = df.loc[outlier_idx, "goals_per90"]
        assert adjusted < raw_value

    def test_missing_mins_played_is_noop(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, enable_limited_sample_training=True, enable_shrinkage=True)
        trainer = Trainer(cfg)
        df = _toy_frame().drop(columns=["mins_played"])
        meta = trainer._apply_shrinkage(df, prior_exclude_mask=df["is_foreign_fallback"])
        assert meta["enabled"] is False
        assert meta["skipped_reason"] == "mins_played missing"


# ── enable_recent_role_features ─────────────────────────────────────────────

class TestRecentRoleFeatures:
    def test_role_features_absent_from_selection_by_default(self, tmp_path: Path) -> None:
        df = _toy_frame()
        numeric, _ = select_features(df)
        role_cols = RoleOpportunityFeatureTransformer().get_feature_names_out()
        assert not any(c in numeric for c in role_cols)

    def test_role_features_reach_selection_when_opted_in(self, tmp_path: Path) -> None:
        df = _toy_frame()
        df = add_role_opportunity_features(
            df, player_col="player_fotmob_id", season_col="season_start"
        )
        role_cols = RoleOpportunityFeatureTransformer().get_feature_names_out()
        numeric, _ = select_features(df, extra_numeric_candidates=role_cols)
        assert any(c in numeric for c in role_cols)

    def test_config_default_is_false(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        assert cfg.enable_recent_role_features is False


# ── enable_breakout_model ────────────────────────────────────────────────────

class TestBreakoutModelShadow:
    def test_skipped_below_minimum_rows(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, enable_breakout_model=True, min_minutes=800, min_minutes_hard=100)
        trainer = Trainer(cfg)
        df = _toy_frame(n_players=3)  # far too few labeled rows
        result = trainer._run_breakout_model(df, ["goals_per90", "goal_assist_per90"])
        assert result["status"] == "skipped"

    def test_trains_and_scores_latest_season_limited_cohort(self, tmp_path: Path) -> None:
        cfg = _cfg(
            tmp_path,
            enable_breakout_model=True,
            min_minutes=800,
            min_minutes_hard=100,
        )
        trainer = Trainer(cfg)
        df = _toy_frame(
            n_players=400,
            seasons=(2019, 2020, 2021, 2022, 2023, 2024),
        )
        result = trainer._run_breakout_model(df, ["goals_per90", "goal_assist_per90"])
        assert result["status"] == "ok"
        assert result["n_train"] > 0
        assert 0.0 <= result["base_rate"] <= 1.0
        if result["predictions"]:
            row = result["predictions"][0]
            assert "breakout_probability" in row
            assert 0.0 <= row["breakout_probability"] <= 1.0

    def test_disabled_flag_is_not_invoked_by_default(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        assert cfg.enable_breakout_model is False


# ── attach_target hard_floor wiring (LIMITED cohort reaches training) ────────

class TestLimitedCohortSurvivesAttachTarget:
    """Regression guard for the wiring bug where attach_target always dropped
    rows below cfg.min_minutes (800), so the LIMITED cohort never reached
    _build_sample_weights / _apply_shrinkage even when the flag was True.
    """

    def _raw_frame(self) -> pd.DataFrame:
        """Synthetic df_raw with one row per minutes bucket."""
        return pd.DataFrame(
            {
                "player_fotmob_id": [1, 2, 3, 4],
                "player_name": ["A", "B", "C", "D"],
                "season_start": [2023, 2023, 2023, 2023],
                "canonical_role": ["FWD", "MID", "DEF", "GK"],
                "mins_played": [50, 300, 600, 900],
                "appearances": [1, 4, 8, 12],
                "goals": [0, 1, 2, 0],
                "is_foreign_fallback": [False, False, False, False],
            }
        )

    def test_legacy_path_drops_limited_cohort(self, tmp_path: Path) -> None:
        """enable_limited_sample_training=False → drop at min_minutes (800)."""
        cfg = _cfg(
            tmp_path,
            enable_limited_sample_training=False,
            min_minutes=800,
            min_minutes_hard=100,
        )
        drop_floor = (
            cfg.min_minutes_hard if cfg.enable_limited_sample_training else cfg.min_minutes
        )
        df = attach_target(
            self._raw_frame(),
            external_csv=None,
            min_minutes=cfg.min_minutes,
            hard_floor=drop_floor,
        )
        mins = set(df["mins_played"].tolist())
        assert 50 not in mins
        assert 300 not in mins
        assert 600 not in mins
        assert 900 in mins

    def test_limited_path_keeps_limited_cohort(self, tmp_path: Path) -> None:
        """enable_limited_sample_training=True → drop only below min_minutes_hard."""
        cfg = _cfg(
            tmp_path,
            enable_limited_sample_training=True,
            min_minutes=800,
            min_minutes_hard=100,
        )
        drop_floor = (
            cfg.min_minutes_hard if cfg.enable_limited_sample_training else cfg.min_minutes
        )
        df = attach_target(
            self._raw_frame(),
            external_csv=None,
            min_minutes=cfg.min_minutes,
            hard_floor=drop_floor,
        )
        mins = set(df["mins_played"].tolist())
        assert 50 not in mins  # below hard floor
        assert 300 in mins
        assert 600 in mins
        assert 900 in mins

    def test_weights_see_limited_rows_when_flag_on(self, tmp_path: Path) -> None:
        """After attach_target with hard_floor, _build_sample_weights must
        produce non-uniform weights for the LIMITED rows that survived.
        """
        cfg = _cfg(
            tmp_path,
            enable_limited_sample_training=True,
            min_minutes=800,
            min_minutes_hard=100,
            weighting_strategy="sqrt",
        )
        trainer = Trainer(cfg)
        drop_floor = cfg.min_minutes_hard
        df = attach_target(
            self._raw_frame(),
            external_csv=None,
            min_minutes=cfg.min_minutes,
            hard_floor=drop_floor,
        )
        weights = trainer._build_sample_weights(df)
        assert weights is not None
        assert len(weights) == 3  # 50 dropped; 300, 600, 900 kept
        # STANDARD row (900) should be full weight; LIMITED rows reduced.
        standard_w = weights[df["mins_played"] == 900].iloc[0]
        limited_w = weights[df["mins_played"] == 300].iloc[0]
        assert standard_w == pytest.approx(1.0)
        assert 0.0 < limited_w < 1.0
