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

# Emit rollout artefacts (effective_config + canary_report) for
# ml-training.yml Phase 6 (WS14, plan §16.1):
    python -m ml.run_pipeline \\
        --emit-effective-config artifacts/effective_config.json \\
        --emit-canary-report artifacts/canary_report.json

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
             "Defaults to Serie A (MLConfig.league_name) when unset — set "
             "the ML_LEAGUE_NAME env var to an empty value to opt into "
             "multi-league training instead.",
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
    parser.add_argument(
        "--evaluate-mantra",
        action="store_true",
        dest="evaluate_mantra",
        help="Run MANTRA 4-pillar evaluation against actuals after training.",
    )
    parser.add_argument(
        "--emit-effective-config",
        default=None,
        dest="emit_effective_config",
        metavar="PATH",
        help=(
            "Write the effective production config (with config_hash, WS16) "
            "to PATH as JSON. Consumed by ml-training.yml Phase 6 to "
            "verify the build matches the deployed rollout state."
        ),
    )
    parser.add_argument(
        "--emit-canary-report",
        default=None,
        dest="emit_canary_report",
        metavar="PATH",
        help=(
            "Write the canary report (limited-cohort safety-net, plan §16.1) "
            "to PATH as JSON. The report exposes anomalies.remaining; "
            "remaining > 0 fails the gate."
        ),
    )
    return parser.parse_args()


# ── Rollout artefacts (WS14, plan §16.1) ─────────────────────────────────────


def _build_effective_config_payload(cfg: MLConfig) -> dict[str, Any]:
    """Build the JSON-serialisable payload for ``effective_config.json``.

    Combines the resolved feature flags (production vs challenger) with
    the MLConfig snapshot, then wraps the result in
    :func:`ml.rollout.config_hash.build_config_bundle` so the artefact
    carries the canonical ``config_hash`` consumed by Phase 6 and Phase 7
    of ``ml-training.yml``.
    """
    # Deferred import: keeps ``--help`` and parse-only paths light, and
    # avoids pulling pydantic-settings indirectly before the env vars
    # are set in ``main``.
    from ml.rollout.config_drift import effective_config_from_mapping
    from ml.rollout.config_hash import build_config_bundle
    from ml.rollout.env_flags import resolve_env_flags

    resolved = resolve_env_flags()
    config_mapping: dict[str, Any] = {
        "production_mode": cfg.reliability_weight_mode,
        "use_new_behavior": any(resolved.production.values()),
        "production_flags": dict(resolved.production),
        "challenger_enabled": any(resolved.challenger.values()),
    }
    eff = effective_config_from_mapping(config_mapping, source="run_pipeline")
    snapshot: dict[str, Any] = {
        "production_mode": eff.production_mode,
        "use_new_behavior": eff.use_new_behavior,
        "production_flags": dict(eff.production_flags),
        "challenger_enabled": eff.challenger_enabled,
        "challenger_flags": dict(resolved.challenger),
        "stages": dict(resolved.stages),
        "source": eff.source,
        "ml_config": {
            "min_minutes": int(cfg.min_minutes),
            "min_minutes_hard": int(cfg.min_minutes_hard),
            "test_seasons": int(cfg.test_seasons),
            "n_clusters": int(cfg.n_clusters),
            "tune": bool(cfg.tune),
            "tune_iter": int(cfg.tune_iter),
            "random_seed": int(cfg.random_seed),
            "weighting_strategy": str(cfg.weighting_strategy),
            "shrinkage_prior_strength": int(cfg.shrinkage_prior_strength),
            "predict_next": bool(cfg.predict_next),
            "league_name": cfg.league_name,
        },
    }
    return build_config_bundle(config=snapshot, extra={"source": "run_pipeline"})


def _write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write *payload* to *path* as UTF-8 JSON, creating parent dirs.

    Atomic-ish: writes to ``<path>.tmp`` then renames, so a partial
    file can never satisfy ``if-no-files-found: error`` downstream.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(target)


def _emit_effective_config(cfg: MLConfig, path: str) -> None:
    log = logging.getLogger(__name__)
    try:
        payload = _build_effective_config_payload(cfg)
        _write_json(path, payload)
        log.info(
            "effective_config written to %s (config_hash=%s)",
            path,
            payload.get("config_hash", "?"),
        )
    except Exception:
        log.exception("Failed to write effective_config to %s", path)


def _emit_canary_report(cfg: MLConfig, path: str) -> None:
    log = logging.getLogger(__name__)
    try:
        from ml.rollout.canary import build_canary_report

        payload = build_canary_report(cfg)
        _write_json(path, payload)
        remaining = payload.get("anomalies", {}).get("remaining", -1)
        log.info(
            "canary_report written to %s (anomalies.remaining=%d, gate=%s)",
            path,
            remaining,
            "PASS" if remaining == 0 else "FAIL",
        )
    except Exception:
        log.exception("Failed to write canary_report to %s", path)


def _emit_rollout_artifacts(cfg: MLConfig, args: argparse.Namespace) -> None:
    """Best-effort emission of WS14 rollout artefacts.

    Both writes are wrapped in their own try/except so a failure in one
    does not suppress the other, and the training run (which already
    succeeded) is not retroactively failed.
    """
    if args.emit_effective_config:
        _emit_effective_config(cfg, args.emit_effective_config)
    if args.emit_canary_report:
        _emit_canary_report(cfg, args.emit_canary_report)


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

    # ── Emit rollout artefacts (WS14 — plan §16.1) ──────────────────────────
    # The two artefacts are independent: each is written only if its
    # path is set, and a failure in either one is logged but does NOT
    # fail the training run (the training already succeeded; the
    # artefacts are for downstream validation).
    if args.emit_effective_config or args.emit_canary_report:
        _emit_rollout_artifacts(cfg, args)

    # ── Evaluate MANTRA (optional) ──────────────────────────────────────────
    if args.evaluate_mantra:
        try:
            from ml.mantra.runner import run_mantra
            from ml.mantra.evaluate import evaluate_mantra_vs_actuals

            season_start = results.get("metadata", {}).get("config", {}).get("season_start")
            if season_start is None:
                latest = results.get("config", {}).get("test_seasons", 1)
                # Infer from predictions
                preds = results.get("predictions", [])
                if preds:
                    season_start = max(p.get("season_start", 0) for p in preds)

            if season_start:
                log.info("Running MANTRA evaluation for season_start=%d", season_start)
                mantra_result = run_mantra(engine, season_start)
                mantra_metrics = evaluate_mantra_vs_actuals(mantra_result, engine, season_start)
                if mantra_metrics:
                    log.info(
                        "MANTRA FP_Mantra-vs-vote ranking: Spearman ρ=%.4f, "
                        "top20 precision=%.4f",
                        mantra_metrics["fp_mantra_vote_spearman"],
                        mantra_metrics["fp_mantra_vote_top20_precision"],
                    )
                else:
                    log.warning(
                        "MANTRA evaluation returned no metrics "
                        "(insufficient data, or degenerate distribution)"
                    )
            else:
                log.warning("Cannot determine season_start for MANTRA evaluation")
        except Exception:
            log.exception("MANTRA evaluation failed (non-fatal)")

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