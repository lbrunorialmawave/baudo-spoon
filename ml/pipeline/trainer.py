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
  "predictions":              [{player_name, season, fantavoto_medio, predicted,
                                 sample_cohort, ml_values_noisy,
                                 predicted_fantavoto_display}, …],
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
  "next_season_predictions":  [{player_name, predicted_next_fantavoto,
                                 sample_cohort, ml_values_noisy,
                                 predicted_next_fantavoto_display}, …],
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
from ..preprocessing.features import (
    _ENVIRONMENTAL_STAT_COLS,
    _PER_90_CANDIDATES,
    engineer_features,
    select_features,
    select_features_rfe,
)
from ..preprocessing.pipeline import build_preprocessor, get_feature_names
from ..preprocessing.role_features import (
    DEFAULT_OPPORTUNITY_WINDOW as _ROLE_OPPORTUNITY_WINDOW,
    DEFAULT_RECENT_WINDOW as _ROLE_RECENT_WINDOW,
    RoleOpportunityFeatureTransformer,
    add_role_opportunity_features,
)
from ..optimizer.models import DEFAULT_BUDGET, ROLE_QUOTAS, TOTAL_SQUAD_SIZE
from ..sample_reliability import (
    compute_sample_weight,
    profile_dataset as profile_sample_dataset,
)
from ..sample_reliability.cohort import (
    COHORT_INSUFFICIENT as _COHORT_INSUFFICIENT,
)
from ..sample_reliability.cohort import (
    COHORT_LIMITED as _COHORT_LIMITED,
)
from ..sample_reliability.cohort import (
    COHORT_STANDARD as _COHORT_STANDARD,
)
from ..sample_reliability.cohort import classify_cohort as _classify_cohort
from ..sample_reliability.output_reliability import attach_output_reliability
from ..sample_reliability.shrinkage import (
    apply_shrinkage as _apply_shrinkage_fn,
    estimate_prior_rate as _estimate_prior_rate,
)
from ..storage.artifact_store import ArtifactStore, R2Config

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

# ── Shrinkage (PR3) ─────────────────────────────────────────────────────────

# Base per-90 columns eligible for shrinkage. Deliberately restricted to
# the *base* "_per90" columns (not "_sap"/"_roll2"/"_delta1" derivatives,
# which are computed from these upstream in engineer_features()) to keep
# the transformation contained and auditable — see plan.md PR3.
_SHRINKAGE_PER90_COLS: frozenset[str] = frozenset(
    f"{c}_per90" for c in _PER_90_CANDIDATES
)

# Minimum STANDARD-cohort rows required (per role group) to estimate a
# shrinkage prior. Below this, the group's prior falls back to the
# dataset-wide STANDARD cohort to avoid a degenerate median-of-few.
_MIN_STANDARD_ROWS_FOR_PRIOR: int = 30

# ── Breakout model (PR7) ─────────────────────────────────────────────────────

# Minimum training rows required to fit the shadow BreakoutClassifier.
_MIN_BREAKOUT_TRAIN_SAMPLES: int = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert a value to a JSON-serialisable type."""
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
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
        # Align production path with deployment env (WS3): ACTIVE flags only.
        # SHADOW challenger flags are observed later via _maybe_write_shadow_artifact.
        try:
            from ml.rollout.env_flags import apply_production_flags_to_config, resolve_env_flags
            resolved = resolve_env_flags()
            apply_production_flags_to_config(cfg, resolved)
            self._rollout_resolved = resolved
        except Exception:  # noqa: BLE001 — never block construction
            self._rollout_resolved = None
        self.cfg = cfg
        self._artifacts_dir = cfg.artifacts_dir
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Unica porta d'ingresso per lettura/scrittura artefatti (cache-aside
        # locale + R2). Vedi design doc "R2 come source of truth" (2026-08-02).
        self._artifact_store = ArtifactStore(
            local_dir=self._artifacts_dir,
            r2_config=R2Config(
                endpoint_url=cfg.r2_endpoint_url,
                access_key_id=cfg.r2_access_key_id,
                secret_access_key=cfg.r2_secret_access_key,
                bucket_name=cfg.r2_bucket_name,
            ),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _artifact(self, filename: str) -> Path:
        return self._artifacts_dir / filename

    def _build_sample_weights(
        self,
        df: pd.DataFrame,
    ) -> pd.Series | None:
        """Return per-row sample weights based on minutes played.

        Behaviour is fully gated by ``cfg.enable_limited_sample_training``.
        When the flag is ``False`` (production default) the function
        returns ``None`` and the trainer falls back to the historical
        uniform-weight behaviour, guaranteeing no behavior change.

        When the flag is ``True`` the function returns a
        ``pd.Series`` aligned with ``df`` containing the weight for
        each row, computed by
        :func:`ml.sample_reliability.weights.compute_sample_weight` with
        the strategy and thresholds from ``MLConfig``.
        """
        if not self.cfg.enable_limited_sample_training:
            return None
        if "mins_played" not in df.columns:
            log.warning(
                "mins_played missing from training frame; "
                "falling back to uniform weights (no limited-sample "
                "down-weighting applied)."
            )
            return None

        minutes = pd.to_numeric(df["mins_played"], errors="coerce")
        weights = minutes.apply(
            lambda m: compute_sample_weight(
                m,
                strategy=self.cfg.weighting_strategy,
                standard_minutes=self.cfg.min_minutes,
                min_minutes_hard=self.cfg.min_minutes_hard,
            )
        )
        weights.index = df.index
        return weights

    def _cohort_profile(self, df: pd.DataFrame) -> dict[str, float | int]:
        """Return a JSON-serialisable summary of cohort counts.

        The summary is also persisted to the artifact store as
        ``cohort_profile.json`` for offline drift analysis.  When the
        ``mins_played`` column is missing the profile is empty.
        """
        if "mins_played" not in df.columns:
            return {
                "n_total": int(len(df)),
                "n_insufficient": 0,
                "n_limited": 0,
                "n_standard": 0,
                "share_insufficient": 0.0,
                "share_limited": 0.0,
                "share_standard": 0.0,
                "min_minutes_hard": int(self.cfg.min_minutes_hard),
                "standard_minutes": int(self.cfg.min_minutes),
            }
        return profile_sample_dataset(
            df,
            minutes_col="mins_played",
            min_minutes_hard=self.cfg.min_minutes_hard,
            standard_minutes=self.cfg.min_minutes,
        )

    def _apply_shrinkage(
        self,
        df: pd.DataFrame,
        *,
        prior_exclude_mask: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Apply per-90 Bayesian shrinkage to *df* in place (PR3).

        No-op (returns ``{"enabled": False}``) unless both
        ``cfg.enable_limited_sample_training`` and ``cfg.enable_shrinkage``
        are ``True`` — mirrors the no-op contract documented on
        ``MLConfig.enable_shrinkage``.

        The population prior for each ``_per90`` column is estimated from
        the STANDARD cohort (``mins_played >= cfg.min_minutes``),
        computed **per role** (``canonical_role``) when that column is
        available, so GK rates never pull outfield rates (and vice
        versa) toward the same prior. Rows in *prior_exclude_mask*
        (typically cross-league fallback rows) never contribute to the
        prior, but the transform is still applied to their features —
        same shrinkage on both the training and inference side, per
        :mod:`ml.sample_reliability.shrinkage`'s no-train/serve-skew
        design.

        Mutates *df* in place and returns a JSON-safe metadata dict
        (also folded into the trainer output's ``sample_reliability``
        section).
        """
        if not (self.cfg.enable_limited_sample_training and self.cfg.enable_shrinkage):
            return {"enabled": False}

        if "mins_played" not in df.columns:
            log.warning(
                "enable_shrinkage=True but 'mins_played' missing; "
                "skipping shrinkage (no-op)."
            )
            return {"enabled": False, "skipped_reason": "mins_played missing"}

        shrink_cols = [c for c in _SHRINKAGE_PER90_COLS if c in df.columns]
        if not shrink_cols:
            log.warning("enable_shrinkage=True but no eligible per-90 columns found.")
            return {"enabled": False, "skipped_reason": "no eligible per90 columns"}

        minutes = pd.to_numeric(df["mins_played"], errors="coerce")
        exclude = (
            prior_exclude_mask.reindex(df.index).fillna(False)
            if prior_exclude_mask is not None
            else pd.Series(False, index=df.index)
        )
        standard_mask_global = (minutes >= self.cfg.min_minutes) & ~exclude

        if "canonical_role" in df.columns:
            role_groups = df.groupby(df["canonical_role"].fillna("UNKNOWN")).groups
        else:
            role_groups = {"ALL": df.index}

        priors: dict[str, dict[str, float]] = {}
        n_adjusted_rows = 0
        for role, idx in role_groups.items():
            idx = pd.Index(idx)
            role_standard_mask = standard_mask_global.loc[idx]
            use_global_fallback = int(role_standard_mask.sum()) < _MIN_STANDARD_ROWS_FOR_PRIOR
            priors[str(role)] = {}
            for col in shrink_cols:
                prior_mask = standard_mask_global if use_global_fallback else (
                    df.index.isin(idx) & standard_mask_global
                )
                prior_rate = _estimate_prior_rate(
                    df.loc[prior_mask, col],
                    minutes=minutes.loc[prior_mask],
                    min_minutes=self.cfg.min_minutes,
                )
                priors[str(role)][col] = prior_rate
                df.loc[idx, col] = _apply_shrinkage_fn(
                    df.loc[idx, col],
                    minutes=minutes.loc[idx],
                    prior_rate=prior_rate,
                    prior_strength=self.cfg.shrinkage_prior_strength,
                )
            n_adjusted_rows += len(idx)

        log.info(
            "Shrinkage applied: %d per-90 column(s) across %d row(s) "
            "(prior_strength=%d, role_groups=%s).",
            len(shrink_cols), n_adjusted_rows,
            self.cfg.shrinkage_prior_strength, list(role_groups.keys()),
        )
        return {
            "enabled": True,
            "columns": shrink_cols,
            "prior_strength": int(self.cfg.shrinkage_prior_strength),
            "priors_by_role": priors,
        }

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
        model_filename = f"{stem}.joblib"
        model_path = self._artifact(model_filename)
        joblib.dump(pipeline, model_path)
        log.info("Model saved: %s", model_path)
        # joblib needs a real local path to dump to; ArtifactStore.save_binary
        # is a no-op copy (source == dest) and just handles the R2 upload.
        self._artifact_store.save_binary(model_path, model_filename)

        # Companion metadata for traceability
        meta = {
            "model_name": model_name,
            "role_prefix": role_prefix,
            "data_hash": data_hash,
            "run_id": self._run_id,
            "artifact": str(model_path.name),
        }
        self._artifact_store.save_json(_json_safe(meta), f"{stem}_meta.json")
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
        # NDJSON is append-only — ArtifactStore.save_json would overwrite it,
        # so we keep the local append above and just delegate the R2 upload
        # of the (already-updated) file via save_binary (no-op copy).
        self._artifact_store.save_binary(log_path, "telemetry_log.ndjson")

    # ── Breakout probability model (PR7, shadow) ──────────────────────────────

    def _run_breakout_model(
        self,
        df_core: pd.DataFrame,
        numeric_features: list[str],
    ) -> dict[str, Any]:
        """Train + evaluate the shadow breakout classifier, and score the
        latest season's LIMITED cohort.

        Purely additive/informational (see call site docstring): returns a
        JSON-safe dict, never raises for expected "not enough data"
        conditions (returns ``status: "skipped"`` instead).
        """
        from ..breakout import (
            build_breakout_labels,
            engineer_breakout_features,
            evaluate_breakout_classifier,
            train_breakout_classifier,
        )

        df_bo = engineer_breakout_features(
            df_core, player_col="player_fotmob_id", season_col="season_start",
        )
        labels_full = build_breakout_labels(
            df_bo,
            player_col="player_fotmob_id",
            season_col="season_start",
            minutes_col="mins_played",
            standard_minutes=self.cfg.min_minutes,
            min_minutes_hard=self.cfg.min_minutes_hard,
        )
        valid = labels_full.notna()
        feature_cols = [c for c in numeric_features if c in df_bo.columns]
        if not feature_cols:
            return {"status": "skipped", "reason": "no numeric features available"}

        bo_df = df_bo.loc[valid]
        bo_labels = labels_full.loc[valid].astype(int)
        base_rate = float(bo_labels.mean()) if len(bo_labels) else 0.0
        log.info(
            "  Breakout dataset: %d labeled row(s) (base_rate=%.3f)",
            len(bo_labels), base_rate,
        )

        if len(bo_df) < _MIN_BREAKOUT_TRAIN_SAMPLES:
            return {
                "status": "skipped",
                "reason": f"only {len(bo_df)} labeled rows (< {_MIN_BREAKOUT_TRAIN_SAMPLES})",
                "n_total": int(len(bo_df)),
                "base_rate": base_rate,
            }

        seasons = sorted(bo_df["season_start"].unique())
        test_season_ids = (
            seasons[-self.cfg.test_seasons:]
            if len(seasons) > self.cfg.test_seasons
            else []
        )
        train_mask = ~bo_df["season_start"].isin(test_season_ids)
        X_train = bo_df.loc[train_mask, feature_cols]
        y_train = bo_labels.loc[train_mask]
        X_test = bo_df.loc[~train_mask, feature_cols]
        y_test = bo_labels.loc[~train_mask]

        if y_train.nunique() < 2 or len(X_train) < _MIN_BREAKOUT_TRAIN_SAMPLES:
            return {
                "status": "skipped",
                "reason": "insufficient class diversity or rows in train split",
                "n_total": int(len(bo_df)),
                "base_rate": base_rate,
            }

        clf = train_breakout_classifier(X_train, y_train, random_seed=self.cfg.random_seed)

        metrics_dict: dict[str, Any] | None = None
        if len(X_test) and y_test.nunique() >= 2:
            metrics = evaluate_breakout_classifier(clf, X_test, y_test)
            metrics_dict = dataclasses.asdict(metrics)
        else:
            log.info("  Breakout test split has <2 classes or is empty; metrics skipped.")

        # Score the latest season's LIMITED cohort (shadow prediction only).
        predictions: list[dict[str, Any]] = []
        if "mins_played" in df_core.columns:
            latest_season = df_core["season_start"].max()
            minutes = pd.to_numeric(df_core["mins_played"], errors="coerce")
            cohort = minutes.apply(
                lambda m: _classify_cohort(
                    m,
                    min_minutes_hard=self.cfg.min_minutes_hard,
                    standard_minutes=self.cfg.min_minutes,
                )
            )
            score_mask = (
                (df_core["season_start"] == latest_season)
                & (cohort == _COHORT_LIMITED)
            )
            if score_mask.any():
                X_score = df_core.loc[score_mask, feature_cols]
                proba = clf.predict_proba(X_score)
                score_cols = [
                    c for c in ("player_fotmob_id", "player_name", "team_name", "season_start")
                    if c in df_core.columns
                ]
                pred_df = df_core.loc[score_mask, score_cols].copy()
                pred_df["breakout_probability"] = proba
                pred_df = pred_df.sort_values("breakout_probability", ascending=False)
                predictions = pred_df.to_dict(orient="records")

        return {
            "status": "ok",
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "base_rate": base_rate,
            "metrics": metrics_dict,
            "feature_cols": feature_cols,
            "predictions": predictions,
        }

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

        # Differentiated imputation: environmental features (team context,
        # quotation signals) use median; event-based features use 0.
        env_feats = [f for f in num_feats if f in _ENVIRONMENTAL_STAT_COLS]
        preprocessor = build_preprocessor(num_feats, cat_feats, environmental_features=env_feats)
        sample_weight = self._build_sample_weights(df_train)
        fitted_pipelines = train_all_models(
            X_train, y_train, preprocessor, self.cfg,
            sample_weight=sample_weight,
        )

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
        return test_metrics, best_name, best_pipe, feature_cols, fitted_pipelines

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
        self._artifact_store.save_json(_json_safe(metadata), f"metadata_{self._run_id}.json")
        log.info("Run metadata saved (deps: %s)", list(metadata["dependencies"].keys()))

        # ── 2. Attach target ──────────────────────────────────────────────────
        log.info("Step 2/12 — Attaching target variable")
        # When limited-sample training is enabled, lower the hard drop floor so
        # the LIMITED cohort (min_minutes_hard .. min_minutes-1) survives and
        # can be weighted / shrunk by the downstream steps. With the flag off
        # behaviour is byte-identical to the legacy pipeline (drop at 800).
        drop_floor = (
            cfg.min_minutes_hard if cfg.enable_limited_sample_training else cfg.min_minutes
        )
        df = attach_target(
            df_raw,
            external_fantavoto_csv,
            cfg.min_minutes,
            hard_floor=drop_floor,
        )

        # ── 3. Feature engineering ────────────────────────────────────────────
        log.info("Step 3/12 — Engineering features")
        df = engineer_features(df, trend_window=2)

        # ── 3a. Role / opportunity features (PR4, opt-in) ─────────────────────
        # No-op unless enable_recent_role_features=True. Computed on the full
        # engineered frame (before the foreign quarantine below) so that
        # cross-league fallback rows get the same features at inference time
        # as training rows — no train/serve skew.
        role_feature_cols: list[str] = []
        if self.cfg.enable_recent_role_features:
            log.info("  enable_recent_role_features=True — adding role/opportunity features")
            df = add_role_opportunity_features(
                df,
                player_col="player_fotmob_id",
                season_col="season_start",
                opportunity_window=_ROLE_OPPORTUNITY_WINDOW,
                recent_window=_ROLE_RECENT_WINDOW,
            )
            role_feature_cols = RoleOpportunityFeatureTransformer().get_feature_names_out()

        # ── 3b. Quarantine cross-league fallback rows ─────────────────────────
        # Neo-arrivi with zero Serie A history (ml/data/loader.py) are
        # inference-only: they must never influence feature selection,
        # training, backtest, or evaluation — only get a prediction (step 8b).
        _foreign_mask = df.get("is_foreign_fallback", pd.Series(False, index=df.index)).fillna(False)

        # ── 3c. Per-90 shrinkage (PR3, opt-in) ─────────────────────────────────
        # No-op unless enable_limited_sample_training AND enable_shrinkage are
        # both True. Priors are estimated only from the STANDARD cohort of
        # df_core (foreign rows excluded via prior_exclude_mask below), but
        # the transform itself is applied to the whole frame — same reasoning
        # as 3a: fallback rows must see the same feature transform at
        # inference as training rows do.
        shrinkage_meta = self._apply_shrinkage(df, prior_exclude_mask=_foreign_mask)

        df_core = df[~_foreign_mask].copy()
        df_foreign = df[_foreign_mask].copy()
        if len(df_foreign):
            log.info(
                "  %d cross-league fallback row(s) quarantined from training/eval",
                len(df_foreign),
            )

        # ── 4. Feature selection ──────────────────────────────────────────────
        log.info("Step 4/12 — Selecting features")
        numeric_features, categorical_features = select_features(
            df_core, extra_numeric_candidates=role_feature_cols
        )

        if not numeric_features:
            raise ValueError(
                "No numeric features available after engineering. "
                "Check that the scraper has collected sufficient stat categories."
            )

        # ── 5. Temporal train/test split ──────────────────────────────────────
        log.info("Step 5/12 — Temporal train/test split")
        df_train, df_test = _temporal_split(df_core, cfg.test_seasons)
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
            gk_test_metrics, best_gk_name, best_gk_pipe, gk_feature_cols, gk_all_pipes = (
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
            out_test_metrics, best_out_name, best_out_pipe, out_feature_cols, out_all_pipes = (
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

            env_feats = [f for f in numeric_features if f in _ENVIRONMENTAL_STAT_COLS]
            preprocessor = build_preprocessor(numeric_features, categorical_features, environmental_features=env_feats)
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

        # ── 8b. Predict for cross-league fallback rows (inference-only) ────────
        # These neo-arrivi were quarantined out of df_core in step 3b — this is
        # the only place they get a prediction, feeding into predictions_df
        # below so season_value/start_probability populate for them too.
        pred_foreign = pd.Series(np.nan, index=df_foreign.index, dtype=float)
        if len(df_foreign):
            if role_partitioned:
                gk_mask_foreign = (
                    df_foreign.get("canonical_role", pd.Series("MID", index=df_foreign.index)) == "GK"
                )
                gk_idx_f = df_foreign.index[gk_mask_foreign]
                out_idx_f = df_foreign.index[~gk_mask_foreign]
                if len(gk_idx_f):
                    pred_foreign.loc[gk_idx_f] = best_gk_pipe.predict(
                        df_foreign.loc[gk_idx_f, gk_feature_cols]
                    )
                if len(out_idx_f):
                    pred_foreign.loc[out_idx_f] = best_out_pipe.predict(
                        df_foreign.loc[out_idx_f, out_feature_cols]
                    )
            else:
                pred_foreign = pd.Series(
                    best_pipe.predict(df_foreign[feature_cols]), index=df_foreign.index
                )
            log.info("  Predicted fantavoto for %d cross-league fallback row(s)", len(df_foreign))

        # ── 9. Backtest ───────────────────────────────────────────────────────
        log.info("Step 9/12 — Walk-forward backtesting")
        from sklearn.base import clone
        bt_result = backtest(
            pipeline=clone(best_pipe),
            df=df_core,
            feature_cols=feature_cols,
            target_col="fantavoto_medio",
            model_name=best_name,
        )

        # Time-series observability: residual drift chart per backtested season
        _plot_residual_drift(
            bt_result,
            str(self._artifact("residual_drift.png")),
        )
        # Upload residual drift chart to R2 (best-effort via ArtifactStore)
        try:
            drift_path = self._artifact("residual_drift.png")
            if drift_path.exists():
                self._artifact_store.save_binary(drift_path, "residual_drift.png")
        except Exception as _drift_exc:  # noqa: BLE001
            log.warning("residual_drift.png upload skipped: %s", _drift_exc)

        # Persist walk-forward residuals for optimizer Monte Carlo (local + R2)
        try:
            from ml.evaluation.residuals_export import (
                build_residuals_payload,
                summarize_residuals,
            )
            residual_rows = list(getattr(bt_result, "residuals", None) or [])
            payload = build_residuals_payload(
                residual_rows,
                run_id=self._run_id,
                model_name=getattr(bt_result, "model_name", "") or best_name,
                source="walkforward_backtest",
                extra_meta={
                    "mean_rmse": getattr(bt_result, "mean_rmse", None),
                    "mean_mae": getattr(bt_result, "mean_mae", None),
                    "n_seasons_tested": len(getattr(bt_result, "season_metrics", []) or []),
                },
            )
            self._artifact_store.save_json(payload, "residuals.json")
            # Also versioned copy for audit
            self._artifact_store.save_json(
                payload, f"residuals_{self._run_id}.json"
            )
            log.info(
                "Residuals exported: %s",
                summarize_residuals(residual_rows),
            )
        except Exception as _res_exc:  # noqa: BLE001 - never fail the pipeline
            log.warning("residuals.json export failed (non-critical): %s", _res_exc)

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

        # Prediction std: cross-model disagreement for each player in df_latest.
        # Uses all fitted pipelines for the primary partition; wrapped in try/except
        # so a missing feature column in an older model never blocks the pipeline.
        try:
            if role_partitioned:
                _all_pipes_latest = out_all_pipes
                _feat_cols_latest = out_feature_cols
                _latest_mask = ~df_latest["canonical_role"].isin(["GK"]) if "canonical_role" in df_latest.columns else pd.Series(True, index=df_latest.index)
            else:
                _all_pipes_latest = fitted_pipelines  # type: ignore[possibly-undefined]
                _feat_cols_latest = feature_cols
                _latest_mask = pd.Series(True, index=df_latest.index)

            _x_latest = df_latest.loc[_latest_mask, _feat_cols_latest]
            if len(_all_pipes_latest) > 1 and not _x_latest.empty:
                _preds_matrix = np.column_stack([
                    pipe.predict(_x_latest) for pipe in _all_pipes_latest.values()
                ])
                _std_series = pd.Series(
                    np.std(_preds_matrix, axis=1, ddof=0),
                    index=_X_latest.index,
                )
                df_latest["prediction_std"] = _std_series.reindex(df_latest.index).fillna(0.0)
            else:
                df_latest["prediction_std"] = 0.0
        except Exception as _std_exc:
            log.warning("prediction_std computation failed (non-critical): %s", _std_exc)
            df_latest["prediction_std"] = 0.0

        cluster_result = run_clustering(df_latest, cfg)
        plot_clusters(
            cluster_result,
            str(self._artifact("cluster_viz.png")),
            rating_col="predicted_fantavoto",
        )
        alternatives = find_low_cost_alternatives(cluster_result)

        # ── 11b. VAR + ESV ────────────────────────────────────────────────────
        log.info("Step 11b — Computing VAR / Expected Surplus Value")
        var_results: list[dict] = []
        try:
            from ml.auction.var import DemandCurve, ReplacementLevel, VAR, VarEngine
            _fc_role_map = {"GK": "P", "DEF": "D", "MID": "C", "FWD": "A"}
            _role_col = "canonical_role" if "canonical_role" in df_latest.columns else None

            # Build player list with Fantacalcio role codes
            var_players: list[dict] = []
            for _, row in df_latest.iterrows():
                raw_role = str(row[_role_col] if _role_col else "MID")
                _sv = row.get("fantapunti_totali")
                _sp = row.get("probabilita_titolarita")
                var_players.append({
                    "player_id": str(row.get("player_fotmob_id", row.get("player_name", "unknown"))),
                    "player_name": str(row.get("player_name", "")),
                    "role": _fc_role_map.get(raw_role, "C"),
                    "projected_score": float(row["predicted_fantavoto"]),
                    "season_value": float(_sv) if pd.notna(_sv) else None,
                    "start_probability": float(_sp) if pd.notna(_sp) else None,
                })

            # Group by role to compute replacement level, then produce VAR + ESV
            from collections import defaultdict as _dd
            by_role: dict[str, list[dict]] = _dd(list)
            for p in var_players:
                by_role[p["role"]].append(p)

            demand = DemandCurve()  # calibrated=False by design
            role_slots = dict(ROLE_QUOTAS)

            for role, role_ps in by_role.items():
                scores = [p["projected_score"] for p in role_ps]
                rl = ReplacementLevel.from_player_pool(role, scores)
                positive_vars = [s - rl.score for s in scores if s > rl.score]
                baseline_var = sum(positive_vars) / len(positive_vars) if positive_vars else 1.0
                budget_per_slot = float(DEFAULT_BUDGET) / TOTAL_SQUAD_SIZE

                for p in role_ps:
                    v = VAR.compute(p["player_id"], role, p["projected_score"], rl)
                    esv_val = (
                        (v.var_score / baseline_var) * budget_per_slot - demand.expected_price(v.var_score)
                        if baseline_var > 0 and v.var_score > 0
                        else demand.base_price - demand.expected_price(v.var_score)
                    )
                    var_results.append({
                        "player_id": p["player_id"],
                        "player_name": p["player_name"],
                        "role": role,
                        "projected_score": round(p["projected_score"], 3),
                        "season_value": round(p["season_value"], 3) if p.get("season_value") is not None else None,
                        "start_probability": round(p["start_probability"], 4) if p.get("start_probability") is not None else None,
                        "replacement_level_score": round(rl.score, 3),
                        "var_score": round(v.var_score, 3),
                        "expected_price": round(demand.expected_price(v.var_score), 2),
                        "esv": round(esv_val, 3),
                        "calibrated": demand.calibrated,
                    })

            var_results.sort(key=lambda r: r["esv"], reverse=True)
            log.info("VAR computed for %d players.", len(var_results))
        except Exception as _var_exc:
            log.warning("VAR computation failed (non-critical): %s", _var_exc)

        # ── 11c. Breakout probability model (shadow, PR7, opt-in) ─────────────
        # No-op unless enable_breakout_model=True. Purely informational: a
        # LogisticRegression P(breakout) is trained on df_core (temporally
        # split like the main model) and scored for the latest season's
        # LIMITED-cohort players. It never feeds back into feature_cols,
        # best_pipe, or predictions_df — the plan is explicit that this stays
        # shadow-only until offline metrics justify promotion (plan.md §48).
        breakout_result: dict[str, Any] = {"enabled": bool(cfg.enable_breakout_model)}
        if cfg.enable_breakout_model:
            log.info("Step 11c — Training shadow breakout classifier")
            try:
                breakout_result.update(
                    self._run_breakout_model(df_core, numeric_features)
                )
            except Exception as _bo_exc:  # noqa: BLE001 — never fail the pipeline
                log.warning("Breakout model training failed (non-critical): %s", _bo_exc)
                breakout_result["status"] = "error"
                breakout_result["error"] = repr(_bo_exc)

        # ── Predict next season (optional) ────────────────────────────────────
        next_season_predictions: list[dict] = []
        if cfg.predict_next:
            log.info(
                "Predict-next mode: re-fitting %s on all %d rows …",
                best_name, len(df_core),
            )
            from sklearn.base import clone as _clone
            full_pipe = _clone(best_pipe)
            full_pipe.fit(df_core[feature_cols], df_core["fantavoto_medio"])

            df_next = df[df["season_start"] == latest_season].copy()
            df_next["predicted_next_fantavoto"] = full_pipe.predict(
                df_next[feature_cols]
            )

            # Label + damp low-sample rows for display (PR9) — never
            # touches the raw predicted_next_fantavoto used elsewhere;
            # only adds sample_cohort / ml_values_noisy / the _display
            # column that the frontend badges off.
            df_next, _next_reliability_meta = attach_output_reliability(
                df_next,
                predicted_col="predicted_next_fantavoto",
                minutes_col="mins_played",
                role_col="canonical_role" if "canonical_role" in df_next.columns else None,
                min_minutes_hard=cfg.min_minutes_hard,
                standard_minutes=cfg.min_minutes,
                prior_strength=cfg.shrinkage_prior_strength,
            limited_ceiling_percentile=cfg.limited_ceiling_percentile,
                exclude_from_prior_mask=df_next.get(
                    "is_foreign_fallback", pd.Series(False, index=df_next.index)
                ).fillna(False),
            )

            next_season_cols = [
                "player_fotmob_id", "player_name", "team_name", "season_start",
                "predicted_next_fantavoto", "predicted_next_fantavoto_display",
                "sample_cohort", "ml_values_noisy",
            ]
            next_season_col = (
                df_next[[c for c in next_season_cols if c in df_next.columns]]
                .sort_values("predicted_next_fantavoto", ascending=False)
                .reset_index(drop=True)
            )
            next_season_predictions = next_season_col.to_dict(orient="records")
            self._artifact_store.save_json(_json_safe(next_season_predictions), "next_season_predictions.json")
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
        predictions_df["is_foreign_fallback"] = False

        # Attach expected_minutes from train data (mins_played in test season)
        # as a simple estimate — the last known minutes for each player.
        if "mins_played" in df_test.columns:
            predictions_df["expected_minutes"] = pd.to_numeric(
                df_test["mins_played"], errors="coerce"
            ).fillna(0).values
        else:
            predictions_df["expected_minutes"] = 0

        # Attach cross-model prediction_std to each test-set player.
        try:
            if role_partitioned:
                _pred_std = pd.Series(0.0, index=df_test.index)
                for _idx, _pipes, _fcols in [
                    (df_test.index[gk_mask_test], gk_all_pipes, gk_feature_cols),
                    (df_test.index[~gk_mask_test], out_all_pipes, out_feature_cols),
                ]:
                    if len(_idx) == 0 or len(_pipes) <= 1:
                        continue
                    _m = np.column_stack([p.predict(df_test.loc[_idx, _fcols]) for p in _pipes.values()])
                    _pred_std.loc[_idx] = np.std(_m, axis=1, ddof=0)
            else:
                if len(fitted_pipelines) > 1:  # type: ignore[possibly-undefined]
                    _m = np.column_stack([p.predict(df_test[feature_cols]) for p in fitted_pipelines.values()])
                    _pred_std = pd.Series(np.std(_m, axis=1, ddof=0), index=df_test.index)
                else:
                    _pred_std = pd.Series(0.0, index=df_test.index)
            predictions_df["prediction_std"] = _pred_std.values
        except Exception as _std_exc2:
            log.warning("prediction_std (test set) failed (non-critical): %s", _std_exc2)
            predictions_df["prediction_std"] = 0.0

        # Fold in cross-league fallback predictions (step 8b) so neo-arrivi
        # with zero Serie A history get season_value/start_probability too —
        # built the same way as predictions_df, minus cross-model ensemble
        # disagreement (prediction_std=0.0: no ensemble computed for this
        # small, low-priority population; a deliberate simplification).
        if len(df_foreign):
            foreign_cols = ["player_fotmob_id", "player_name", "team_name", "season_start"]
            if "canonical_role" in df_foreign.columns:
                foreign_cols.append("canonical_role")
            foreign_predictions_df = df_foreign[foreign_cols].copy()
            foreign_predictions_df["fantavoto_medio"] = df_foreign["fantavoto_medio"].values
            foreign_predictions_df["predicted_fantavoto"] = pred_foreign.values
            foreign_predictions_df["expected_minutes"] = (
                pd.to_numeric(df_foreign.get("mins_played"), errors="coerce").fillna(0).values
                if "mins_played" in df_foreign.columns else 0
            )
            foreign_predictions_df["prediction_std"] = float(os.environ.get("ML_FOREIGN_FALLBACK_PREDICTION_STD", "1.5"))
            foreign_predictions_df["is_foreign_fallback"] = True
            predictions_df = pd.concat(
                [predictions_df, foreign_predictions_df], ignore_index=True, sort=False
            )

        # Label + damp low-sample rows for display (PR9) — see
        # ``ml.sample_reliability.output_reliability``. Runs on the full
        # predictions_df (test-set STANDARD/LIMITED rows + any folded-in
        # foreign-fallback rows); foreign-fallback rows are still
        # classified/damped but excluded from prior estimation since
        # they're already a separate, cross-league signal.
        predictions_df, output_reliability_meta = attach_output_reliability(
            predictions_df,
            predicted_col="predicted_fantavoto",
            minutes_col="expected_minutes",
            role_col="canonical_role" if "canonical_role" in predictions_df.columns else None,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
            prior_strength=cfg.shrinkage_prior_strength,
            limited_ceiling_percentile=cfg.limited_ceiling_percentile,
            exclude_from_prior_mask=predictions_df.get(
                "is_foreign_fallback", pd.Series(False, index=predictions_df.index)
            ).fillna(False),
        )

        # Derive season-value targets from predicted rating × predicted appearances.
        # The derivation lives in ``ml.domain.predictions`` so the MANTRA runner
        # and the optimizer pool read the same numbers from the same source.
        from ml.domain.predictions import derive_season_value_columns
        derive_season_value_columns(predictions_df)

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
            "var_results": var_results,
            "sample_reliability": {
                "enabled": bool(cfg.enable_limited_sample_training),
                "weighting_strategy": cfg.weighting_strategy,
                "shrinkage_enabled": bool(cfg.enable_shrinkage),
                "shrinkage_prior_strength": int(cfg.shrinkage_prior_strength),
                "shrinkage": shrinkage_meta,
                "recent_role_features_enabled": bool(cfg.enable_recent_role_features),
                "recent_role_features": role_feature_cols,
                "breakout_model_enabled": bool(cfg.enable_breakout_model),
                "cohort_profile": self._cohort_profile(df_core),
                "output_reliability": output_reliability_meta,
            },
            "breakout_model": breakout_result,
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
        output_safe = _json_safe(output)
        self._artifact_store.save_json(output_safe, f"results_{self._run_id}.json")
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
        self._artifact_store.save_json(output_safe, "results_latest.json")

        self._persist_metrics_to_db(output)

        # Shadow rollout artifact (WS3): when CHALLENGER env flags are set,
        # emit baseline vs challenger decision-score comparison without
        # changing the production decision path used above.
        self._maybe_write_shadow_artifact(output)

        log.info("=" * 60)
        log.info("Pipeline complete.  Results in %s", self._artifacts_dir)
        return output

    def _maybe_write_shadow_artifact(self, output: dict) -> None:
        """If SHADOW challenger flags are set, write comparison artifact (WS3).

        Production decisions already used the ACTIVE-only config. This only
        observes the challenger path on the emitted prediction rows.
        """
        try:
            from ml.rollout.env_flags import resolve_env_flags
            from ml.rollout.shadow_artifacts import write_shadow_artifact
        except ImportError:
            return
        resolved = resolve_env_flags()
        if not resolved.any_challenger():
            return
        preds = output.get("predictions") or []
        if not preds:
            return
        players = []
        for row in preds:
            if not isinstance(row, dict):
                continue
            players.append(
                {
                    "player_id": str(row.get("player_fotmob_id") or row.get("player_name") or ""),
                    "role": row.get("canonical_role") or row.get("role") or "?",
                    "minutes": row.get("mins_played") or row.get("minutes"),
                    "projected_score": row.get("predicted") or row.get("predicted_fantavoto_display") or 0.0,
                }
            )
        out_path = Path(self._artifacts_dir) / f"shadow_comparison_{self._run_id}.json"
        try:
            write_shadow_artifact(
                out_path,
                players,
                baseline_mode="bucket",
                challenger_mode="continuous",
                meta={
                    "run_id": self._run_id,
                    "stages": resolved.stages,
                    "production_flags": resolved.production,
                    "challenger_flags": resolved.challenger,
                },
            )
            log.info("Shadow comparison artifact written to %s", out_path)
        except Exception as exc:  # noqa: BLE001 — never fail the main pipeline
            log.warning("Failed to write shadow artifact: %s", exc)

    def _persist_metrics_to_db(self, output: dict) -> None:
        """Write run metadata and metrics to Postgres. Non-fatal: logs on failure."""

        import os

        db_url = os.environ.get("ML_DATABASE_URL") or os.environ.get("API_DATABASE_URL")
        if not db_url:
            log.warning("ML_DATABASE_URL not set — skipping DB metrics persist")
            return

        # Trainer is sync; normalise asyncpg DSN to plain psycopg2.
        sync_url = (
            db_url
            .replace("postgresql+asyncpg://", "postgresql://")
            .replace("postgres+asyncpg://", "postgresql://")
        )

        try:
            engine = sa.create_engine(sync_url, future=True)
            with engine.begin() as conn:
                conn.execute(sa.text("""
                    INSERT INTO model_runs
                        (run_id, model_name, trained_at, season_start,
                         training_seasons, hyperparams, dependencies, git_commit)
                    VALUES
                        (:run_id, :model_name, NOW(), :season_start,
                         CAST(:training_seasons AS jsonb), CAST(:hyperparams AS jsonb),
                         CAST(:dependencies AS jsonb), :git_commit)
                    ON CONFLICT (run_id) DO NOTHING
                """), {
                    "run_id": output["run_id"],
                    "model_name": output["best_model"],
                    "season_start": output.get("metadata", {}).get("config", {}).get("season_start"),
                    "training_seasons": json.dumps(output.get("config", {}).get("test_seasons", [])),
                    "hyperparams": json.dumps(output.get("metadata", {}).get("best_params", {})),
                    "dependencies": json.dumps(output.get("metadata", {}).get("dependencies", {})),
                    "git_commit": _git_commit(),
                })

                best = next(
                    (m for m in output.get("model_comparison", [])
                     if m.get("model") == output["best_model"]),
                    None,
                )
                if best:
                    for metric in ("rmse", "mae", "r2"):
                        if best.get(metric) is not None:
                            conn.execute(sa.text("""
                                INSERT INTO model_metrics
                                    (run_id, metric_name, metric_value, split)
                                VALUES (:run_id, :metric_name, :metric_value, 'test')
                                ON CONFLICT DO NOTHING
                            """), {"run_id": output["run_id"],
                                   "metric_name": metric,
                                   "metric_value": best[metric]})

                bt = output.get("backtest", {})
                for metric, col in (("rmse", "mean_rmse"), ("mae", "mean_mae"), ("r2", "mean_r2")):
                    if bt.get(col) is not None:
                        conn.execute(sa.text("""
                            INSERT INTO model_metrics
                                (run_id, metric_name, metric_value, split)
                            VALUES (:run_id, :metric_name, :metric_value, 'backtest')
                            ON CONFLICT DO NOTHING
                        """), {"run_id": output["run_id"],
                               "metric_name": metric,
                               "metric_value": bt[col]})

                _check_drift(conn, output["run_id"], output["best_model"])

            log.info("Metrics persisted to DB for run %s", output["run_id"])
        except Exception as exc:
            log.error("Failed to persist metrics to DB (non-fatal): %s", exc, exc_info=True)


# ── Module-level helpers used by Trainer._persist_metrics_to_db ──────────────


def _git_commit() -> str | None:
    """Return the short git commit hash, or None if git is unavailable."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _check_drift(conn: sa.Connection, run_id: str, model_name: str, threshold_pct: float = 10.0) -> None:
    """Mark run as 'degraded' if test RMSE exceeds the 5-run moving average by threshold_pct."""
    baseline_row = conn.execute(sa.text("""
        SELECT AVG(metric_value) AS baseline
        FROM (
            SELECT mm.metric_value
            FROM model_metrics mm
            JOIN model_runs mr ON mr.run_id = mm.run_id
            WHERE mr.model_name = :model_name
              AND mm.metric_name = 'rmse'
              AND mm.split = 'test'
              AND mr.run_id != :run_id
            ORDER BY mr.trained_at DESC
            LIMIT 5
        ) recent
    """), {"model_name": model_name, "run_id": run_id}).fetchone()

    current = conn.execute(sa.text("""
        SELECT metric_value FROM model_metrics
        WHERE run_id = :run_id AND metric_name = 'rmse' AND split = 'test'
        LIMIT 1
    """), {"run_id": run_id}).scalar()

    if not (baseline_row and baseline_row.baseline and current):
        return

    pct_change = (current - baseline_row.baseline) / baseline_row.baseline * 100
    if pct_change > threshold_pct:
        conn.execute(sa.text(
            "UPDATE model_runs SET status = 'degraded' WHERE run_id = :run_id"
        ), {"run_id": run_id})
        conn.execute(sa.text("""
            INSERT INTO model_drift_alerts
                (run_id, metric_name, current_value, baseline_value, pct_change, threshold_pct)
            VALUES (:run_id, 'rmse', :current, :baseline, :pct, :threshold)
        """), {
            "run_id": run_id,
            "current": current,
            "baseline": baseline_row.baseline,
            "pct": pct_change,
            "threshold": threshold_pct,
        })
        log.warning(
            "Model drift detected for run %s: RMSE %.4f vs baseline %.4f (+%.1f%%)",
            run_id, current, baseline_row.baseline, pct_change,
        )