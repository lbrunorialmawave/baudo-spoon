#!/usr/bin/env python
"""Fantasy-football ML pipeline — command-line entry point.

Usage examples
--------------
# Minimal (all leagues, approximate target):
    ML_DATABASE_URL="postgresql+psycopg2://fbref:pass@localhost:5432/fbref" \\
    python -m ml.run_pipeline

# Serie A only, with real fantavoto CSV and hyperparameter tuning:
    ML_DATABASE_URL="..." \\
    python -m ml.run_pipeline \\
        --league "Serie A" \\
        --fantavoto-csv path/to/fantavoto.csv \\
        --tune \\
        --clusters 8

# Auto-select K via Silhouette method:
    python -m ml.run_pipeline --clusters -1

# Structured JSON logs for ELK/Splunk:
    python -m ml.run_pipeline --json-logs

# Docker (reads ML_DATABASE_URL from environment, set in docker-compose):
    docker compose run --rm api python -m ml.run_pipeline --league "Serie A"

Exit codes
----------
0 — success
1 — pipeline error
2 — configuration error
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Structured JSON logging ───────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for ELK/Splunk integration.

    Each log record is serialised as a single-line JSON object with fields:
    ``timestamp`` (ISO-8601 UTC), ``level``, ``logger``, ``message``, and
    optionally ``exception`` / ``stack_info``.

    Example output::

        {"timestamp": "2024-01-01T12:00:00.123456+00:00", "level": "INFO",
         "logger": "ml.pipeline.trainer", "message": "Step 1/12 — …"}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_obj["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(log_obj, ensure_ascii=False)


def _configure_logging(level: str, json_logs: bool = False) -> None:
    """Configure root logger with either JSON or human-readable format."""
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


# ── Database resiliency ───────────────────────────────────────────────────────

def _create_engine_with_retry(
    db_url: str,
    max_attempts: int = 5,
    base_delay: float = 1.0,
) -> Any:
    """Create a SQLAlchemy engine with exponential backoff on connection failure.

    Probes the connection with a lightweight ``SELECT 1`` after each creation
    attempt so transient network glitches (e.g. PostgreSQL cold-start in
    Docker Compose, serverless wake-up latency) are transparently retried.

    Back-off schedule: delay = base_delay × 2^(attempt - 1)
    Example (base_delay=1.0): 1 s, 2 s, 4 s, 8 s, 16 s → 5 attempts max.

    Args:
        db_url: SQLAlchemy connection URL string.
        max_attempts: Maximum number of connection attempts before raising.
        base_delay: Initial retry delay in seconds (doubles each attempt).

    Returns:
        A connected :class:`sqlalchemy.Engine` instance.

    Raises:
        RuntimeError: If all *max_attempts* are exhausted.
    """
    import sqlalchemy as sa

    log = logging.getLogger(__name__)
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            engine = sa.create_engine(db_url, pool_pre_ping=True)
            # Lightweight probe — verifies the engine can actually reach the DB
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            log.info("Database engine ready (attempt %d/%d).", attempt, max_attempts)
            return engine
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * math.pow(2, attempt - 1)
            log.warning(
                "DB connection failed (attempt %d/%d): %s — retrying in %.1f s …",
                attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to connect to the database after {max_attempts} attempts."
    ) from last_exc


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ml.run_pipeline",
        description="Predict fantasy-football (fantavoto) ratings per player-season.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--league",
        default=None,
        metavar="NAME",
        help="Filter to a specific league (partial match, e.g. 'Serie A'). "
             "Defaults to all leagues.",
    )
    parser.add_argument(
        "--fantavoto-csv",
        default=None,
        metavar="PATH",
        dest="fantavoto_csv",
        help="Path to external CSV with actual fantavoto_medio values. "
             "If omitted, the target is approximated from FotMob stats.",
    )
    parser.add_argument(
        "--test-seasons",
        type=int,
        default=1,
        dest="test_seasons",
        metavar="N",
        help="Number of most-recent seasons held out as the test set.",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=800,
        dest="min_minutes",
        metavar="N",
        help="Minimum minutes played per season to include a player.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        metavar="K",
        help="Number of KMedoids clusters.  Pass -1 to auto-select K via "
             "the Silhouette method (evaluates k ∈ [2, 10]).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV hyperparameter tuning (slower).",
    )
    parser.add_argument(
        "--predict-next",
        action="store_true",
        dest="predict_next",
        help=(
            "After evaluation, re-fit on all data and predict next-season "
            "fantavoto from the most-recent season's stats. "
            "Output saved to next_season_predictions.json."
        ),
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=30,
        dest="tune_iter",
        metavar="N",
        help="Number of parameter combinations in RandomizedSearchCV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        metavar="DIR",
        help="Directory for artefacts (models, plots, JSON). "
             "Defaults to ml/artifacts/.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        dest="json_logs",
        help="Emit logs as JSON objects (one per line) for ELK/Splunk integration.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level, json_logs=args.json_logs)
    log = logging.getLogger(__name__)

    # ── Validate database URL ─────────────────────────────────────────────────
    db_url = os.environ.get("ML_DATABASE_URL") or os.environ.get("API_DATABASE_URL")
    if not db_url:
        log.error(
            "Database URL not set. "
            "Set ML_DATABASE_URL (or API_DATABASE_URL) environment variable."
        )
        return 2

    # ── Build config overriding env with CLI flags ────────────────────────────
    # Set env vars before importing MLConfig so pydantic-settings picks them up.
    os.environ["ML_DATABASE_URL"] = db_url
    os.environ["ML_LOG_LEVEL"] = args.log_level
    os.environ["ML_RANDOM_SEED"] = str(args.seed)
    os.environ["ML_TEST_SEASONS"] = str(args.test_seasons)
    os.environ["ML_MIN_MINUTES"] = str(args.min_minutes)
    os.environ["ML_N_CLUSTERS"] = str(args.clusters)
    os.environ["ML_TUNE"] = "true" if args.tune else "false"
    os.environ["ML_TUNE_ITER"] = str(args.tune_iter)
    os.environ["ML_PREDICT_NEXT"] = "true" if args.predict_next else "false"
    if args.league:
        os.environ["ML_LEAGUE_NAME"] = args.league
    if args.output_dir:
        os.environ["ML_ARTIFACTS_DIR"] = args.output_dir

    # Deferred import so env vars are set before pydantic-settings resolves.
    from ml.config import MLConfig
    from ml.pipeline.trainer import Trainer

    cfg = MLConfig()
    _configure_logging(cfg.log_level, json_logs=args.json_logs)

    fantavoto_csv = Path(args.fantavoto_csv) if args.fantavoto_csv else None
    if fantavoto_csv and not fantavoto_csv.exists():
        log.error("fantavoto CSV not found: %s", fantavoto_csv)
        return 2

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        engine = _create_engine_with_retry(db_url)
        trainer = Trainer(cfg)
        results = trainer.run(external_fantavoto_csv=fantavoto_csv, engine=engine)
    except Exception:
        log.exception("Pipeline failed with an unhandled exception.")
        return 1

    # ── Print summary to stdout ───────────────────────────────────────────────
    summary = {
        "best_model": results["best_model"],
        "role_partitioned": results.get("role_partitioned", False),
        "test_metrics": next(
            (r for r in results["model_comparison"] if r["model"] == results["best_model"]),
            {},
        ),
        "role_metrics": results.get("role_metrics", {}),
        "backtest": {
            k: results["backtest"][k]
            for k in ("mean_rmse", "mean_mae", "mean_r2")
        },
        "n_predictions": len(results["predictions"]),
        "n_clusters": results["clustering_stats"]["n_clusters"],
        "silhouette": results["clustering_stats"]["silhouette"],
        "n_low_cost_alternatives": len(results["low_cost_recommendations"]),
        "data_hash": results.get("metadata", {}).get("data_hash", ""),
        "artifacts_dir": str(cfg.artifacts_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
