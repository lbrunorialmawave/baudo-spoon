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

    # ── League filter ─────────────────────────────────────────────────────────
    # When None, all leagues in the database are used.
    league_name: str | None = None

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


# Singleton — imported by all submodules.
settings = MLConfig()
