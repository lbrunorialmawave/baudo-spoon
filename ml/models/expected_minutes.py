"""Expected Minutes Model — independent submodel.

Predicts expected_minutes for the upcoming season, given:
    - Historical minutes/appearances
    - Age
    - Injury/suspension history (as absence ratio)
    - Rotation indicators (competition for starting position)
    - Team competition context (European fixtures proxy)
    - Coach continuity

This model is trained independently from the main fantavoto ensemble.
Its output (expected_minutes, confidence) feeds ExpectedMinutesFeature,
which is one of the inputs to the Phase 3 stacking ensemble.

Validation: TimeSeriesSplit walk-forward backtest (own metrics, separate from
the main pipeline's backtest). Never shares CV folds with the main ensemble.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

# ── Input feature spec ────────────────────────────────────────────────────────

# All column names after canonicalize_columns() has been applied.
_FEATURE_COLS: list[str] = [
    "mins_played",          # lagged (previous season) — set by _build_features()
    "appearances",          # lagged appearances
    "age",                  # player age at season start (if available)
    "absence_ratio",        # fraction of possible minutes missed (derived)
    "team_strength_score",  # team context
    "is_top_team",
    "role_code",            # ordinal role encoding
    "season_idx",           # temporal trend
]

# Minimum training rows required to fit the model.
_MIN_TRAIN_ROWS: int = 30


@dataclass(frozen=True)
class ExpectedMinutesResult:
    """Output for a single player-season prediction."""
    player_fotmob_id: str
    season_start: int
    expected_minutes: float
    confidence: float   # in [0, 1]; derived from prediction interval width


class ExpectedMinutesModel:
    """Submodel predicting expected minutes for the next season.

    Fit on lagged (previous season) stats → target (current season minutes).
    Validated independently with TimeSeriesSplit.

    Usage::

        model = ExpectedMinutesModel(random_seed=42)
        model.fit(df_historical)
        predictions = model.predict(df_latest_season)
        backtest_result = model.backtest(df_historical)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self._pipeline: Optional[Pipeline] = None
        self._feature_cols_used: list[str] = []

    # ── Feature construction ──────────────────────────────────────────────────

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Construct the feature set for the expected minutes model.

        Applies one-season lag to playing-time columns so that the model
        predicts next-season minutes from previous-season stats.
        Creates absence_ratio = 1 - (mins_played / (appearances * 90)).

        Args:
            df: Player-season DataFrame sorted by (player_fotmob_id, season_start).
                Must contain mins_played or appearances.

        Returns:
            DataFrame with lagged features and derived columns.
        """
        df = df.sort_values(["player_fotmob_id", "season_start"]).copy()

        # Lag playing-time inputs: model must only see PREVIOUS season data
        for col in ("mins_played", "appearances", "team_strength_score", "is_top_team"):
            if col in df.columns:
                df[f"{col}_lag1"] = df.groupby("player_fotmob_id")[col].shift(1)

        # Absence ratio (fraction of possible minutes missed in previous season)
        if "mins_played_lag1" in df.columns and "appearances_lag1" in df.columns:
            possible = (df["appearances_lag1"].clip(lower=1) * 90.0)
            df["absence_ratio"] = (
                1.0 - (df["mins_played_lag1"].clip(lower=0) / possible.clip(lower=1))
            ).clip(0.0, 1.0)
        else:
            df["absence_ratio"] = np.nan

        return df

    @staticmethod
    def _resolve_feature_cols(df: pd.DataFrame) -> list[str]:
        """Return the subset of _FEATURE_COLS (+ lagged variants) present in df."""
        lag_variants = [
            "mins_played_lag1", "appearances_lag1",
            "team_strength_score_lag1", "is_top_team_lag1",
        ]
        candidates = _FEATURE_COLS + lag_variants
        # Prefer lagged versions when both exist
        available = [c for c in candidates if c in df.columns and df[c].notna().any()]
        # Deduplicate: if *_lag1 present, skip the unlagged version
        deduped: list[str] = []
        lagged = {c.replace("_lag1", "") for c in available if c.endswith("_lag1")}
        for c in available:
            base = c.replace("_lag1", "")
            if not c.endswith("_lag1") and base in lagged:
                continue  # skip unlagged when lagged version present
            deduped.append(c)
        return deduped

    @staticmethod
    def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]),
                    feature_cols,
                )
            ],
            remainder="drop",
        )

    # ── Fit / predict ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ExpectedMinutesModel":
        """Train on historical player-season data.

        Target: mins_played (current season).
        Features: lagged stats from previous season.

        Args:
            df: Feature DataFrame after build_features() has been applied.
                Must contain mins_played as target.

        Returns:
            self (fitted).
        """
        df_feat = self.build_features(df)
        feature_cols = self._resolve_feature_cols(df_feat)

        if "mins_played" not in df_feat.columns:
            raise ValueError("ExpectedMinutesModel.fit requires 'mins_played' target column.")

        # Drop rows without target or without any feature data
        valid_mask = df_feat["mins_played"].notna()
        df_feat = df_feat[valid_mask].copy()

        if len(df_feat) < _MIN_TRAIN_ROWS:
            raise ValueError(
                f"ExpectedMinutesModel requires at least {_MIN_TRAIN_ROWS} training rows, "
                f"got {len(df_feat)}."
            )

        X = df_feat[feature_cols]
        y = df_feat["mins_played"].clip(lower=0.0)

        preprocessor = self._build_preprocessor(feature_cols)
        estimator = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=self.random_seed,
        )
        self._pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        self._pipeline.fit(X, y)
        self._feature_cols_used = feature_cols
        log.info(
            "ExpectedMinutesModel fitted on %d rows, %d features.",
            len(df_feat), len(feature_cols),
        )
        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> list[ExpectedMinutesResult]:
        """Predict expected minutes for each row in df.

        Args:
            df: Player-season DataFrame (after build_features()).

        Returns:
            List of ExpectedMinutesResult ordered by df row order.
        """
        if self._pipeline is None:
            raise RuntimeError("ExpectedMinutesModel must be fitted before predict().")

        df_feat = self.build_features(df)
        X = df_feat[self._feature_cols_used].reindex(
            columns=self._feature_cols_used, fill_value=np.nan
        )
        raw_pred = self._pipeline.predict(X).clip(min=0.0)

        # Confidence: based on how complete the feature set is per row
        # (fraction of feature columns that are non-NaN per row)
        non_null_frac = X.notna().mean(axis=1).values

        results: list[ExpectedMinutesResult] = []
        for i, (_, row) in enumerate(df_feat.iterrows()):
            results.append(ExpectedMinutesResult(
                player_fotmob_id=str(row.get("player_fotmob_id", "")),
                season_start=int(row.get("season_start", 0)),
                expected_minutes=float(raw_pred[i]),
                confidence=float(non_null_frac[i]),
            ))
        return results

    # ── Backtest ──────────────────────────────────────────────────────────────

    def backtest(
        self,
        df: pd.DataFrame,
        n_splits: int = 3,
    ) -> dict[str, object]:
        """Walk-forward backtest using TimeSeriesSplit.

        This is the model's own independent backtest — separate from the
        main pipeline's backtest. It uses TimeSeriesSplit to respect
        chronological ordering of seasons.

        Args:
            df: Full historical player-season DataFrame.
            n_splits: Number of temporal CV folds (default 3).

        Returns:
            Dict with mean_rmse, mean_mae, season_metrics list.
        """
        df_feat = self.build_features(df)
        feature_cols = self._resolve_feature_cols(df_feat)

        if "mins_played" not in df_feat.columns:
            raise ValueError("backtest requires 'mins_played' column.")

        valid = df_feat["mins_played"].notna()
        df_feat = df_feat[valid].sort_values("season_start").copy()

        if len(df_feat) < _MIN_TRAIN_ROWS * 2:
            log.warning(
                "Too few rows (%d) for meaningful backtest; "
                "returning empty metrics.", len(df_feat)
            )
            return {"mean_rmse": float("nan"), "mean_mae": float("nan"), "season_metrics": []}

        X = df_feat[feature_cols]
        y = df_feat["mins_played"].clip(lower=0.0).values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        season_metrics: list[dict[str, object]] = []
        rmses: list[float] = []
        maes: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(X_train) < _MIN_TRAIN_ROWS:
                log.debug("Fold %d: too few training rows (%d); skipping.", fold_idx, len(X_train))
                continue

            preprocessor = self._build_preprocessor(feature_cols)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.05,
                    max_leaf_nodes=15, random_state=self.random_seed,
                )),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test).clip(min=0.0)

            rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            mae = float(np.mean(np.abs(preds - y_test)))
            rmses.append(rmse)
            maes.append(mae)

            test_seasons = df_feat.iloc[test_idx]["season_start"].unique().tolist()
            season_metrics.append({
                "fold": fold_idx,
                "test_seasons": test_seasons,
                "n_test": len(test_idx),
                "rmse": rmse,
                "mae": mae,
            })
            log.info(
                "ExpectedMinutesModel backtest fold %d: RMSE=%.1f, MAE=%.1f",
                fold_idx, rmse, mae,
            )

        return {
            "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "mean_mae": float(np.mean(maes)) if maes else float("nan"),
            "season_metrics": season_metrics,
        }
