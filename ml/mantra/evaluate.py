"""Evaluate MANTRA predictions against actuals and register in model_runs/model_metrics."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import sqlalchemy as sa

log = logging.getLogger(__name__)


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()[:12]
    except Exception:
        return None


def evaluate_mantra_vs_actuals(
    mantra_result: dict[str, Any],
    engine: sa.Engine,
    season_start: int,
) -> dict[str, float] | None:
    """Compare MANTRA VR predictions to actual fantavoto_medio from DB.

    Computes RMSE, MAE, R2 and persists to model_runs/model_metrics with
    model_name='mantra-4pillar' so it appears in /model-metrics/compare.

    Returns metrics dict or None if insufficient data.
    """
    players = mantra_result.get("players", [])
    if not players:
        log.warning("No MANTRA players to evaluate")
        return None

    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT fantacalcio_id, vote_avg
            FROM player_season_aggregates psa
            JOIN player_id_map pim ON pim.player_fotmob_id = psa.fantacalcio_id::text
            WHERE psa.season_start = :season_start AND vote_avg IS NOT NULL
        """), {"season_start": season_start}).fetchall()

        if not rows:
            rows = conn.execute(sa.text("""
                SELECT pq.fantacalcio_id, pss.vote_avg
                FROM player_quotations pq
                JOIN player_id_map pim ON pim.fantacalcio_id = pq.fantacalcio_id
                    AND pim.season_start = pq.season_start
                LEFT JOIN player_season_aggregates pss
                    ON pss.fantacalcio_id = pim.player_fotmob_id::bigint
                    AND pss.season_start = pq.season_start
                WHERE pq.season_start = :season_start AND pss.vote_avg IS NOT NULL
            """), {"season_start": season_start}).fetchall()

    actuals = {int(r[0]): float(r[1]) for r in rows}
    if not actuals:
        log.warning("No actual ratings found for season %d", season_start)
        return None

    y_true = []
    y_pred = []
    for p in players:
        fid = p.get("fantacalcio_id")
        vr = p.get("VR")
        if fid in actuals and vr is not None:
            y_true.append(actuals[fid])
            y_pred.append(float(vr))

    if len(y_true) < 10:
        log.warning("Too few matched players (%d) for MANTRA evaluation", len(y_true))
        return None

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    log.info("MANTRA evaluation: RMSE=%.4f MAE=%.4f R2=%.4f (n=%d)", rmse, mae, r2, len(y_true))

    run_id = f"mantra-{season_start}-{uuid.uuid4().hex[:8]}"
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("""
                INSERT INTO model_runs
                    (run_id, model_name, trained_at, season_start, git_commit, status)
                VALUES
                    (:run_id, 'mantra-4pillar', NOW(), :season_start, :git_commit, 'completed')
                ON CONFLICT (run_id) DO NOTHING
            """), {
                "run_id": run_id,
                "season_start": season_start,
                "git_commit": _git_commit(),
            })

            for metric_name, value in metrics.items():
                conn.execute(sa.text("""
                    INSERT INTO model_metrics
                        (run_id, metric_name, metric_value, split)
                    VALUES (:run_id, :metric_name, :metric_value, 'test')
                    ON CONFLICT DO NOTHING
                """), {"run_id": run_id, "metric_name": metric_name, "metric_value": value})

        log.info("MANTRA metrics persisted as run %s", run_id)
    except Exception as exc:
        log.error("Failed to persist MANTRA metrics (non-fatal): %s", exc)

    return metrics
