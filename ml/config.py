from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


class MLConfig(BaseSettings):
    """Runtime configuration for the fantasy-football ML pipeline.

    All values can be overridden via environment variables prefixed with ML_,
    or via a .env file in the project root.

    Assumptions:
    - The database schema is the FotMob platform defined in db/init.sql.
    - ``player_season_stats`` stores one row per (player, season, stat_category).
    - The target variable (fantavoto_medio) is either supplied as an external
      CSV or approximated from goal/assist/card stats.
    """

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL (psycopg2 dialect)",
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Temporal split ────────────────────────────────────────────────────────
    # Hold out the N most-recent seasons as the test set.  All earlier seasons
    # form the training window.  This preserves temporal ordering.
    test_seasons: int = 1

    # ── Data quality ─────────────────────────────────────────────────────────
    # Players with fewer than this number of minutes played are excluded to
    # avoid noisy target estimates from small samples.
    min_minutes: int = 800

    # ── Low-sample / breakout modelling (plan.md, PR1–PR8) ──────────────────
    # Lower eligibility cutoff for the LIMITED cohort (100..799 minutes).
    # Training/inference of LIMITED rows is gated by feature flags below;
    # this value is metadata, not a hard exclusion filter.
    min_minutes_hard: int = 100

    # Master switch for the low-sample weighting pipeline.  When False the
    # trainer behaves exactly as before (no row of minutes < ``min_minutes``
    # reaches training).  When True, LIMITED rows may be included with a
    # reduced weight and per-90 shrinkage applied.
    enable_limited_sample_training: bool = False
    # Master switch for per-90 shrinkage (PR3).  No-op when
    # ``enable_limited_sample_training`` is False.
    enable_shrinkage: bool = False
    # Master switch for the breakout probability model (PR7).  Kept
    # disabled until offline experiments validate the dataset.
    enable_breakout_model: bool = False

    # Sample-weighting strategy.  Supported: "constant", "linear", "sqrt",
    # "bucketed".  See ``ml.sample_reliability.weights.compute_sample_weight``.
    weighting_strategy: str = "sqrt"
    # Number of pseudo-observations the population prior contributes
    # during per-90 shrinkage.  See ``apply_shrinkage``.
    shrinkage_prior_strength: int = 300

    # ── League filter ─────────────────────────────────────────────────────────
    # Defaults to Serie A: once foreign leagues are scraped (for MANTRA's
    # cross-league neo-arrivo fallback, see ml/mantra/runner.py), a training
    # run without this filter would silently start pooling their rows into
    # the Serie A predictor's training set. Pass None explicitly to opt into
    # multi-league training deliberately.
    league_name: str | None = "Serie A"

    # Cross-league fallback for neo-arrivi (players with zero Serie A history):
    # append their most recent season from ANY league (player_latest_stats_any_league,
    # migration 018) at inference time only — never used for model training/fitting.
    # Mirrors MANTRA's stats_from_foreign_league fallback (ml/mantra/runner.py).
    include_foreign_fallback: bool = True

    # ── Clustering ────────────────────────────────────────────────────────────
    n_clusters: int = 6
    # Fraction of variance retained by PCA before KMeans.
    pca_variance_threshold: float = 0.90

    # ── SHAP ──────────────────────────────────────────────────────────────────
    # Subsample size for SHAP TreeExplainer (speed vs. accuracy trade-off).
    shap_sample_size: int = 300

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    # When True, run RandomizedSearchCV; when False, use sensible defaults.
    tune: bool = False
    # Number of parameter combinations to try in RandomizedSearchCV.
    tune_iter: int = 30
    # Number of TimeSeriesSplit folds.
    cv_folds: int = 3

    # ── Future-season inference ───────────────────────────────────────────────
    # When True, re-fit the best model on ALL available data after evaluation
    # and apply it to the most-recent season's features to produce a ranked
    # list of next-season predictions (saved to next_season_predictions.json).
    predict_next: bool = False

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Output ────────────────────────────────────────────────────────────────
    artifacts_dir: Path = ARTIFACTS_DIR

    # ── Artifact storage (Cloudflare R2) ────────────────────────────────────
    r2_endpoint_url: str | None = Field(default=None, description="https://<account_id>.r2.cloudflarestorage.com")
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket_name: str = Field(default="baudo-spoon-ml-artifacts")


# Singleton — imported by all submodules.
settings = MLConfig()
