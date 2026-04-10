"""End-to-end training and inference pipeline orchestrator.

Full pipeline steps:
1. Build run metadata (hardware, deps, data hash) and connect to DB.
2. Load raw player + team stats.
3. Attach target variable (fantavoto_medio from CSV or approximation).
4. Engineer features (per-90, SAP, rolling averages, deltas).
5. Temporal train/test split (hold out most-recent N seasons).
6. **Role-Partitioned Training**: GK and Outfield players are trained on
   separate sub-pipelines with role-appropriate feature sets.
7. Evaluate all models; report metrics separately for GK and Outfield.
8. Select the best model (lowest RMSE within each role; Outfield model
   used as the primary pipeline for backtest/explainability).
9. Run walk-forward backtesting on the best Outfield model.
10. Compute explainability (SHAP + feature importance) for the best model.
11. Run KMedoids clustering with PCA on the latest season's data.
12. Find low-cost player alternatives.
13. Persist all artefacts and return a structured output dict.

Output dict (also serialised to JSON in the artifacts directory):
{
  "run_id":                   "20240101_120000",
  "best_model":               "xgboost",
  "role_partitioned":         true,
  "predictions":              [{player_name, season, fantavoto_medio, predicted}, …],
  "model_comparison":         [{model, rmse, mae, r2}, …],
  "role_metrics": {
    "gk":       {"ridge": {rmse, mae, r2}, …},
    "outfield":  {"ridge": {rmse, mae, r2}, …},
  },
  "feature_importance":       [{feature, importance}, …],
  "backtest":                 {mean_rmse, mean_mae, mean_r2, season_metrics: […]},
  "player_clusters":          [{player_name, cluster_id, pca_0, pca_1, …}, …],
  "low_cost_recommendations": [{top_player_name, alt_player_name, …}, …],
  "clustering_stats":         {n_clusters, silhouette, inertia, pca_explained_variance},
  "next_season_predictions":  [{player_name, predicted_next_fantavoto}, …],
  "metadata":                 {run_id, hardware, dependencies, data_hash, config},
  "config":                   {…},
}
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import sqlalchemy as sa

from ..clustering.kmeans import find_low_cost_alternatives, plot_clusters, run_clustering
from ..config import MLConfig
from ..data.loader import load_raw_data
from ..data.target import attach_target
from ..evaluation.explainability import (
    compute_permutation_importance,
    compute_shap_values,
    compute_tree_feature_importance,
    plot_feature_importance,
    plot_shap_summary,
)
from ..evaluation.metrics import (
    SplitMetrics,
    backtest,
    build_comparison_table,
    evaluate_on_test,
)
from ..models.regression import train_all_models
from ..preprocessing.features import engineer_features, select_features, select_features_rfe
from ..preprocessing.pipeline import build_preprocessor, get_feature_names

log = logging.getLogger(__name__)

# ── Role-partition constants ───────────────────────────────────────────────────

# Minimum GK training samples required to fork a dedicated sub-pipeline.
_MIN_GK_TRAIN_SAMPLES: int = 20

# Attacking features that are meaningless / always-zero for GKs.
_GK_EXCLUDE_FEATURES: frozenset[str] = frozenset([
    "goals_per90", "goal_assist_per90",
    "total_scoring_att_per90", "ontarget_scoring_att_per90",
    "big_chance_created_per90", "big_chance_missed_per90",
    "total_att_assist_per90", "won_contest_per90",
    "goals_per90_roll2", "goal_assist_per90_roll2",
    "total_scoring_att_per90_roll2",
    "goals_per90_delta1", "goal_assist_per90_delta1",
    "goals_per90_sap", "goal_assist_per90_sap",
    "total_scoring_att_per90_sap", "ontarget_scoring_att_per90_sap",
])

# GK-specific features that are meaningless for outfielders.
_OUTFIELD_EXCLUDE_FEATURES: frozenset[str] = frozenset([
    "saves_per90", "_goals_prevented_per90",
    "clean_sheet_per90", "goals_conceded_per90",
    "saves_per90_roll2", "_goals_prevented_per90_roll2",
    "saves_per90_delta1", "_goals_prevented_per90_delta1",
    "saves_per90_sap", "_goals_prevented_per90_sap",
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert a value to a JSON-serialisable type."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj))
    return obj


def _temporal_split(
    df: pd.DataFrame,
    test_seasons: int,
    season_col: str = "season_start",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test by holding out the N most-recent seasons."""
    seasons = sorted(df[season_col].unique())
    if len(seasons) <= test_seasons:
        raise ValueError(
            f"Need at least {test_seasons + 1} seasons to hold out {test_seasons} "
            f"for testing; only {len(seasons)} available."
        )
    test_season_ids = seasons[-test_seasons:]
    train_mask = ~df[season_col].isin(test_season_ids)
    log.info(
        "Temporal split: train seasons=%s | test seasons=%s",
        seasons[:-test_seasons],
        test_season_ids,
    )
    return df[train_mask].copy(), df[~train_mask].copy()


def _filter_features_for_role(
    numeric_features: list[str],
    categorical_features: list[str],
    role: str,
) -> tuple[list[str], list[str]]:
    """Return feature lists appropriate for *role* ('GK' or 'OUTFIELD')."""
    exclude = _GK_EXCLUDE_FEATURES if role == "GK" else _OUTFIELD_EXCLUDE_FEATURES
    filtered_numeric = [f for f in numeric_features if f not in exclude]
    return filtered_numeric, list(categorical_features)


def _compute_data_hash(df: pd.DataFrame) -> str:
    """Return a SHA-256 hex digest of the DataFrame for auditability."""
    h = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    )
    return f"sha256:{h.hexdigest()}"


def _gather_metadata(
    run_id: str,
    cfg: MLConfig,
    data_hash: str,
) -> dict[str, Any]:
    """Collect hardware specs, dependency versions, and config for metadata.json."""
    _DEPS = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("scikit-learn-extra", "sklearn_extra"),
        ("xgboost", "xgboost"),
        ("shap", "shap"),
        ("joblib", "joblib"),
        ("pydantic", "pydantic"),
    ]
    dep_versions: dict[str, str] = {}
    for dep_name, import_name in _DEPS:
        try:
            mod = importlib.import_module(import_name)
            dep_versions[dep_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            dep_versions[dep_name] = "not installed"

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hardware": {
            "hostname": platform.node(),
            "cpu_count": os.cpu_count() or 1,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "dependencies": dep_versions,
        "data_hash": data_hash,
        "config": {
            "test_seasons": cfg.test_seasons,
            "min_minutes": cfg.min_minutes,
            "league_name": cfg.league_name,
            "random_seed": cfg.random_seed,
            "n_clusters": cfg.n_clusters,
            "tune": cfg.tune,
            "predict_next": cfg.predict_next,
        },
    }


def _plot_residual_drift(
    bt_result: Any,
    output_path: str,
) -> None:
    """Save a time-series RMSE/MAE plot across all backtested seasons.

    Visualises prediction-error drift over chronological test seasons so that
    performance decay (or improvement) from concept drift can be detected
    early.  Dashed horizontal lines mark the walk-forward mean for reference.

    The plot is saved as ``residual_drift.png`` in the artifacts directory
    alongside the existing cluster and SHAP visualisations.

    Args:
        bt_result: :class:`~evaluation.metrics.BacktestResult` from
            :func:`~evaluation.metrics.backtest`.
        output_path: Absolute path to write the PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not bt_result.season_metrics:
        log.warning("_plot_residual_drift: no backtest seasons available; skipping plot.")
        return

    seasons = [s["test_season"] for s in bt_result.season_metrics]
    rmses = [s["rmse"] for s in bt_result.season_metrics]
    maes = [s["mae"] for s in bt_result.season_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(seasons, rmses, marker="o", linewidth=2, label="RMSE", color="tab:red")
    ax.plot(seasons, maes, marker="s", linewidth=2, linestyle="--", label="MAE", color="tab:blue")
    ax.axhline(
        bt_result.mean_rmse,
        color="tab:red", linestyle=":", alpha=0.55,
        label=f"Mean RMSE = {bt_result.mean_rmse:.3f}",
    )
    ax.axhline(
        bt_result.mean_mae,
        color="tab:blue", linestyle=":", alpha=0.55,
        label=f"Mean MAE = {bt_result.mean_mae:.3f}",
    )
    ax.set_xlabel("Test Season (backtested)")
    ax.set_ylabel("Prediction Error")
    ax.set_title(
        f"Residual Drift Across Backtested Seasons — {bt_result.model_name}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(seasons)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Residual drift plot saved to %s", output_path)


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates the full ML pipeline from DB to JSON output.

    Usage::

        trainer = Trainer(cfg)
        results = trainer.run(external_fantavoto_csv=None)
    """

    def __init__(self, cfg: MLConfig) -> None:
        self.cfg = cfg
        self._artifacts_dir = cfg.artifacts_dir
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _artifact(self, filename: str) -> Path:
        return self._artifacts_dir / filename

    def _save_json(self, data: Any, filename: str) -> None:
        path = self._artifact(filename)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)
        log.info("Saved %s", path)

    def _save_model(
        self,
        pipeline: Any,
        model_name: str,
        data_hash: str,
        role_prefix: str = "",
    ) -> Path:
        """Persist *pipeline* using the ``{model_name}_{hash}_{ts}`` convention.

        The data hash is injected into the artefact filename and a companion
        ``*_meta.json`` file for downstream traceability.

        Args:
            pipeline: Fitted sklearn Pipeline to serialise.
            model_name: Human-readable model identifier.
            data_hash: SHA-256 hash string (``sha256:...``) of the training data.
            role_prefix: Optional prefix for role-partitioned models (e.g. ``"gk_"``).

        Returns:
            Path to the saved ``.joblib`` file.
        """
        hash_short = data_hash.replace("sha256:", "")[:8]
        stem = f"{role_prefix}{model_name}_{hash_short}_{self._run_id}"
        model_path = self._artifact(f"{stem}.joblib")
        joblib.dump(pipeline, model_path)
        log.info("Model saved: %s", model_path)

        # Companion metadata for traceability
        meta = {
            "model_name": model_name,
            "role_prefix": role_prefix,
            "data_hash": data_hash,
            "run_id": self._run_id,
            "artifact": str(model_path.name),
        }
        self._save_json(meta, f"{stem}_meta.json")
        return model_path

    def _export_telemetry(
        self,
        data_hash: str,
        model_metrics: dict[str, Any],
        clustering_stats: dict[str, Any],
    ) -> None:
        """Append a telemetry record to the timeseries performance log.

        Exports RMSE, MAE, R², clustering Inertia, and Silhouette score to a
        newline-delimited JSON log (``telemetry_log.ndjson``) that is suitable
        for time-series ingestion and dashboarding.

        Args:
            data_hash: SHA-256 data hash for this run.
            model_metrics: Dict with keys ``rmse``, ``mae``, ``r2``.
            clustering_stats: Dict with keys ``inertia``, ``silhouette``.
        """
        record: dict[str, Any] = {
            "run_id": self._run_id,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "data_hash": data_hash,
            "metrics": {
                "RMSE": model_metrics.get("rmse"),
                "MAE": model_metrics.get("mae"),
                "R2": model_metrics.get("r2"),
                "Inertia": clustering_stats.get("inertia"),
                "Silhouette": clustering_stats.get("silhouette"),
            },
        }
        log_path = self._artifact("telemetry_log.ndjson")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record)) + "\n")
        log.info("Telemetry appended to %s", log_path)

    # ── Role-partitioned sub-pipeline ─────────────────────────────────────────

    def _run_role_pipeline(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        role: str,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> tuple[dict, str, Any, list[str]]:
        """Train, evaluate and return results for a single role partition.

        Args:
            df_train: Training rows for this role.
            df_test: Test rows for this role.
            role: 'GK' or 'OUTFIELD'.
            numeric_features: Global numeric feature list (will be filtered).
            categorical_features: Global categorical features.

        Returns:
            Tuple of (test_metrics_dict, best_model_name, best_pipeline, feature_cols).
        """
        num_feats, cat_feats = _filter_features_for_role(
            numeric_features, categorical_features, role
        )

        # Apply RFE to prune collinear numeric features; preserves at least 70%
        num_feats = select_features_rfe(
            X_train=df_train[num_feats + cat_feats],
            y_train=df_train["fantavoto_medio"],
            numeric_features=num_feats,
            n_features_fraction=0.70,
        )

        feature_cols = num_feats + cat_feats
        log.info(
            "[%s] features: %d numeric + %d categorical",
            role, len(num_feats), len(cat_feats),
        )

        X_train = df_train[feature_cols]
        y_train = df_train["fantavoto_medio"]
        X_test = df_test[feature_cols]
        y_test = df_test["fantavoto_medio"]

        preprocessor = build_preprocessor(num_feats, cat_feats)
        fitted_pipelines = train_all_models(X_train, y_train, preprocessor, self.cfg)

        test_metrics: dict[str, SplitMetrics] = {}
        for name, pipe in fitted_pipelines.items():
            m = evaluate_on_test(
                pipe, X_test, y_test, model_name=f"{role.lower()}_{name}"
            )
            test_metrics[name] = m
            log.info(
                "[%s] %s → RMSE=%.4f, MAE=%.4f, R²=%.4f",
                role, name, m.rmse, m.mae, m.r2,
            )

        comparison_df = build_comparison_table(test_metrics)
        best_name = comparison_df.iloc[0]["model"]
        best_pipe = fitted_pipelines[best_name]
        log.info(
            "[%s] best model: %s (RMSE=%.4f)",
            role, best_name, comparison_df.iloc[0]["rmse"],
        )
        return test_metrics, best_name, best_pipe, feature_cols

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        external_fantavoto_csv: Optional[Path] = None,
        engine: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline and return the results dict.

        Args:
            external_fantavoto_csv: Optional path to a CSV file with actual
                fantavoto data.  When None, the target is approximated.
            engine: Optional pre-built SQLAlchemy engine.  When provided, this
                engine is used directly (e.g. one built with exponential backoff
                by :func:`~run_pipeline._create_engine_with_retry`).  When None,
                a plain engine is created from ``cfg.database_url``.

        Returns:
            Nested dict with predictions, metrics, cluster info, and
            explainability insights.
        """
        cfg = self.cfg
        log.info("=" * 60)

        # ── 1. Connect + load data ─────────────────────────────────────────────
        log.info("Step 1/12 — Connecting to database and loading data")
        if engine is None:
            engine = sa.create_engine(cfg.database_url)
        df_raw = load_raw_data(engine, cfg)

        # Compute data hash before any transformations for auditability
        data_hash = _compute_data_hash(df_raw)
        log.info("Data hash: %s (%d rows)", data_hash, len(df_raw))

        # ── 1b. Build and persist run metadata ────────────────────────────────
        metadata = _gather_metadata(self._run_id, cfg, data_hash)
        self._save_json(metadata, f"metadata_{self._run_id}.json")
        log.info("Run metadata saved (deps: %s)", list(metadata["dependencies"].keys()))

        # ── 2. Attach target ──────────────────────────────────────────────────
        log.info("Step 2/12 — Attaching target variable")
        df = attach_target(df_raw, external_fantavoto_csv, cfg.min_minutes)

        # ── 3. Feature engineering ────────────────────────────────────────────
        log.info("Step 3/12 — Engineering features")
        df = engineer_features(df, trend_window=2)

        # ── 4. Feature selection ──────────────────────────────────────────────
        log.info("Step 4/12 — Selecting features")
        numeric_features, categorical_features = select_features(df)

        if not numeric_features:
            raise ValueError(
                "No numeric features available after engineering. "
                "Check that the scraper has collected sufficient stat categories."
            )

        # ── 5. Temporal train/test split ──────────────────────────────────────
        log.info("Step 5/12 — Temporal train/test split")
        df_train, df_test = _temporal_split(df, cfg.test_seasons)
        log.info("  Train: %d rows | Test: %d rows", len(df_train), len(df_test))

        # ── 6. Role partition ─────────────────────────────────────────────────
        log.info("Step 6/12 — Role-partitioned sub-pipeline (GK vs Outfield)")
        gk_mask_train = df_train.get("canonical_role", pd.Series("MID", index=df_train.index)) == "GK"
        gk_mask_test = df_test.get("canonical_role", pd.Series("MID", index=df_test.index)) == "GK"
        n_gk_train = int(gk_mask_train.sum())

        role_partitioned = n_gk_train >= _MIN_GK_TRAIN_SAMPLES
        if role_partitioned:
            log.info(
                "  GK partition: %d train / %d test rows",
                n_gk_train, int(gk_mask_test.sum()),
            )
        else:
            log.warning(
                "  Only %d GK training rows (threshold %d); "
                "skipping GK-specific sub-pipeline.",
                n_gk_train, _MIN_GK_TRAIN_SAMPLES,
            )

        # ── 7. Train & evaluate — role-partitioned ────────────────────────────
        log.info("Step 7/12 — Training regression models")

        role_metrics: dict[str, dict[str, Any]] = {}

        if role_partitioned:
            # ── GK sub-pipeline ───────────────────────────────────────────────
            gk_test_metrics, best_gk_name, best_gk_pipe, gk_feature_cols = (
                self._run_role_pipeline(
                    df_train[gk_mask_train].reset_index(drop=True),
                    df_test[gk_mask_test].reset_index(drop=True),
                    "GK",
                    numeric_features,
                    categorical_features,
                )
            )
            role_metrics["gk"] = {
                name: m.as_dict() for name, m in gk_test_metrics.items()
            }

            # ── Outfield sub-pipeline ─────────────────────────────────────────
            out_test_metrics, best_out_name, best_out_pipe, out_feature_cols = (
                self._run_role_pipeline(
                    df_train[~gk_mask_train].reset_index(drop=True),
                    df_test[~gk_mask_test].reset_index(drop=True),
                    "OUTFIELD",
                    numeric_features,
                    categorical_features,
                )
            )
            role_metrics["outfield"] = {
                name: m.as_dict() for name, m in out_test_metrics.items()
            }

            # Primary model = best outfield (most rows; used for backtest/SHAP)
            best_name = best_out_name
            best_pipe = best_out_pipe
            feature_cols = out_feature_cols

            # Combined comparison table from outfield metrics (primary)
            comparison_df = build_comparison_table(out_test_metrics)

        else:
            # Unified pipeline (no GK fork)
            numeric_features = select_features_rfe(
                X_train=df_train[numeric_features + categorical_features],
                y_train=df_train["fantavoto_medio"],
                numeric_features=numeric_features,
                n_features_fraction=0.70,
            )
            feature_cols = numeric_features + categorical_features
            X_train = df_train[feature_cols]
            y_train = df_train["fantavoto_medio"]
            X_test = df_test[feature_cols]
            y_test = df_test["fantavoto_medio"]

            preprocessor = build_preprocessor(numeric_features, categorical_features)
            fitted_pipelines = train_all_models(X_train, y_train, preprocessor, cfg)

            test_metrics_unified: dict[str, SplitMetrics] = {}
            for name, pipe in fitted_pipelines.items():
                m = evaluate_on_test(pipe, X_test, y_test, model_name=name)
                test_metrics_unified[name] = m

            comparison_df = build_comparison_table(test_metrics_unified)
            best_name = comparison_df.iloc[0]["model"]
            best_pipe = fitted_pipelines[best_name]
            log.info(
                "Best model: %s (RMSE=%.4f)", best_name, comparison_df.iloc[0]["rmse"]
            )

        log.info("\n%s", comparison_df.to_string(index=False))

        # ── 8. Build combined test predictions ────────────────────────────────
        log.info("Step 8/12 — Assembling combined test predictions")
        pred_test = pd.Series(np.nan, index=df_test.index, dtype=float)

        if role_partitioned:
            gk_idx = df_test.index[gk_mask_test]
            out_idx = df_test.index[~gk_mask_test]
            if len(gk_idx):
                pred_test.loc[gk_idx] = best_gk_pipe.predict(
                    df_test.loc[gk_idx, gk_feature_cols]
                )
            if len(out_idx):
                pred_test.loc[out_idx] = best_out_pipe.predict(
                    df_test.loc[out_idx, out_feature_cols]
                )
        else:
            pred_test = pd.Series(
                best_pipe.predict(df_test[feature_cols]), index=df_test.index
            )

        # ── 9. Backtest ───────────────────────────────────────────────────────
        log.info("Step 9/12 — Walk-forward backtesting")
        from sklearn.base import clone
        bt_result = backtest(
            pipeline=clone(best_pipe),
            df=df,
            feature_cols=feature_cols,
            target_col="fantavoto_medio",
            model_name=best_name,
        )

        # Time-series observability: residual drift chart per backtested season
        _plot_residual_drift(
            bt_result,
            str(self._artifact("residual_drift.png")),
        )

        # ── 10. Explainability ────────────────────────────────────────────────
        log.info("Step 10/12 — Computing explainability")
        feat_names_transformed = get_feature_names(
            best_pipe.named_steps["preprocessor"]
        )

        feat_imp_df: Optional[pd.DataFrame] = None
        model = best_pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            feat_imp_df = compute_tree_feature_importance(best_pipe, feat_names_transformed)
        else:
            feat_imp_df = compute_permutation_importance(
                best_pipe,
                df_test.loc[~gk_mask_test, feature_cols] if role_partitioned else df_test[feature_cols],
                df_test.loc[~gk_mask_test, "fantavoto_medio"] if role_partitioned else df_test["fantavoto_medio"],
                feature_names=feat_names_transformed,
                random_seed=cfg.random_seed,
            )

        plot_feature_importance(
            feat_imp_df,
            str(self._artifact(f"feature_importance_{best_name}.png")),
            model_name=best_name,
        )

        shap_result = compute_shap_values(
            best_pipe,
            df_train.loc[~gk_mask_train, feature_cols] if role_partitioned else df_train[feature_cols],
            feature_names=feat_names_transformed,
            sample_size=cfg.shap_sample_size,
            random_seed=cfg.random_seed,
        )
        if shap_result is not None:
            shap_vals, X_sample_transformed = shap_result
            plot_shap_summary(
                shap_vals,
                X_sample_transformed,
                feat_names_transformed,
                str(self._artifact(f"shap_{best_name}.png")),
                model_name=best_name,
            )

        # ── 11. Clustering ────────────────────────────────────────────────────
        log.info("Step 11/12 — Running player clustering")
        latest_season = df["season_start"].max()
        df_latest = df[df["season_start"] == latest_season].copy()
        df_latest["predicted_fantavoto"] = best_pipe.predict(
            df_latest[feature_cols]
        )

        cluster_result = run_clustering(df_latest, cfg)
        plot_clusters(
            cluster_result,
            str(self._artifact("cluster_viz.png")),
            rating_col="predicted_fantavoto",
        )
        alternatives = find_low_cost_alternatives(cluster_result)

        # ── Predict next season (optional) ────────────────────────────────────
        next_season_predictions: list[dict] = []
        if cfg.predict_next:
            log.info(
                "Predict-next mode: re-fitting %s on all %d rows …",
                best_name, len(df),
            )
            from sklearn.base import clone as _clone
            full_pipe = _clone(best_pipe)
            full_pipe.fit(df[feature_cols], df["fantavoto_medio"])

            df_next = df[df["season_start"] == latest_season].copy()
            df_next["predicted_next_fantavoto"] = full_pipe.predict(
                df_next[feature_cols]
            )
            next_season_col = (
                df_next[
                    ["player_fotmob_id", "player_name", "team_name",
                     "season_start", "predicted_next_fantavoto"]
                ]
                .sort_values("predicted_next_fantavoto", ascending=False)
                .reset_index(drop=True)
            )
            next_season_predictions = next_season_col.to_dict(orient="records")
            self._save_json(next_season_predictions, "next_season_predictions.json")
            log.info(
                "Next-season predictions saved: %d players",
                len(next_season_predictions),
            )

        # ── 12. Assemble output ───────────────────────────────────────────────
        log.info("Step 12/12 — Assembling output")

        y_test_vals = df_test["fantavoto_medio"]
        cols = ["player_fotmob_id", "player_name", "team_name", "season_start"]
        if "canonical_role" in df_test.columns:
            cols.append("canonical_role")
        predictions_df = df_test[cols].copy()
        predictions_df["fantavoto_medio"] = y_test_vals.values
        predictions_df["predicted_fantavoto"] = pred_test.values

        output: dict[str, Any] = {
            "run_id": self._run_id,
            "best_model": best_name,
            "role_partitioned": role_partitioned,
            "predictions": predictions_df.to_dict(orient="records"),
            "model_comparison": comparison_df.to_dict(orient="records"),
            "role_metrics": role_metrics,
            "feature_importance": feat_imp_df.to_dict(orient="records") if feat_imp_df is not None else [],
            "backtest": {
                "mean_rmse": bt_result.mean_rmse,
                "mean_mae": bt_result.mean_mae,
                "mean_r2": bt_result.mean_r2,
                "season_metrics": bt_result.season_metrics,
            },
            "player_clusters": (
                lambda df: df.rename(columns={"predicted_fantavoto": "fantavoto_medio"})[
                    [c for c in ["player_fotmob_id", "player_name", "team_name",
                                 "canonical_role", "fantavoto_medio", "cluster_id",
                                 "pca_0", "pca_1"] if c in df.columns]
                ].to_dict(orient="records")
            )(cluster_result.df),
            "low_cost_recommendations": [
                dataclasses.asdict(a) for a in alternatives
            ],
            "clustering_stats": {
                "n_clusters": cluster_result.n_clusters_used,
                "silhouette": cluster_result.silhouette,
                "inertia": cluster_result.inertia,
                "pca_explained_variance": cluster_result.explained_variance,
            },
            "next_season_predictions": next_season_predictions,
            "metadata": metadata,
            "config": {
                "test_seasons": cfg.test_seasons,
                "min_minutes": cfg.min_minutes,
                "league_name": cfg.league_name,
                "random_seed": cfg.random_seed,
                "tune": cfg.tune,
            },
        }

        # Persist
        self._save_json(output, f"results_{self._run_id}.json")
        self._save_model(best_pipe, best_name, data_hash)
        if role_partitioned:
            self._save_model(best_gk_pipe, best_gk_name, data_hash, role_prefix="gk_")
        log.info("Model(s) saved to artifacts/")

        # Telemetry export (timeseries-ready NDJSON log)
        best_metrics = comparison_df.iloc[0].to_dict() if len(comparison_df) > 0 else {}
        self._export_telemetry(
            data_hash=data_hash,
            model_metrics={
                "rmse": best_metrics.get("rmse"),
                "mae": best_metrics.get("mae"),
                "r2": best_metrics.get("r2"),
            },
            clustering_stats={
                "inertia": cluster_result.inertia,
                "silhouette": cluster_result.silhouette,
            },
        )

        # Latest snapshot for easy reference
        latest_path = self._artifact("results_latest.json")
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(output), f, indent=2, ensure_ascii=False)

        log.info("=" * 60)
        log.info("Pipeline complete.  Results in %s", self._artifacts_dir)
        return output
