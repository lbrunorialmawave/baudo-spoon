"""Evaluate MANTRA predictions against actuals and register in model_runs/model_metrics."""

from __future__ import annotations

import logging
import uuid
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
    """Compare MANTRA FP_Mantra predictions to actual fantavoto_medio from DB.

    ``FP_Mantra`` (0-100 merit score) and ``vote_avg`` (typically 5-8) are
    on unrelated absolute scales, so this does NOT compute RMSE/MAE/R² —
    those would be meaningless here (see git history for the incident this
    fixes: an earlier version compared them directly and produced
    RMSE > 100, R² in the -100000s). Instead it evaluates *ranking*
    agreement, which is scale-invariant and also the property that actually
    matters for MANTRA's use case (ordering players by value, not
    predicting their literal average vote):

    - ``spearman``: Spearman rank correlation between FP_Mantra and
      vote_avg across all matched players (-1..1, higher is better).
    - ``top20_precision``: fraction of the actual top-20%-by-vote_avg
      players that MANTRA also places in its own top 20% by FP_Mantra.

    Persists to model_runs/model_metrics with model_name='mantra-4pillar'
    so it appears in /model-metrics/compare, under metric names prefixed
    ``fp_mantra_vote_`` so they're never confused with the vote-scale
    RMSE/MAE/R² reported by the regular fantavoto regression models.

    Returns metrics dict or None if insufficient data.

    Notes
    -----
    ``player_season_aggregates.fantacalcio_id`` is a *misnomer*: the view
    (migration 010) aliases ``player_season_stats.player_fotmob_id`` as
    ``fantacalcio_id``. Both sides of the join to ``player_id_map.player_fotmob_id``
    are therefore BIGINT FotMob ids — never cast to text.
    """
    players = mantra_result.get("players", [])
    if not players:
        log.warning("No MANTRA players to evaluate")
        return None

    # Primary path: aggregates view keyed by FotMob id (aliased as fantacalcio_id)
    # → map to real fantacalcio_id via player_id_map.
    primary_sql = sa.text(
        """
        SELECT pim.fantacalcio_id, psa.vote_avg
        FROM player_season_aggregates psa
        JOIN player_id_map pim
          ON pim.player_fotmob_id = psa.fantacalcio_id
         AND pim.season_start = psa.season_start
        WHERE psa.season_start = :season_start
          AND psa.vote_avg IS NOT NULL
          AND pim.fantacalcio_id IS NOT NULL
        """
    )

    # Fallback: start from quotations (true fantacalcio_id) and join stats via map.
    fallback_sql = sa.text(
        """
        SELECT pq.fantacalcio_id, psa.vote_avg
        FROM player_quotations pq
        JOIN player_id_map pim
          ON pim.fantacalcio_id = pq.fantacalcio_id
         AND pim.season_start = pq.season_start
        JOIN player_season_aggregates psa
          ON psa.fantacalcio_id = pim.player_fotmob_id
         AND psa.season_start = pq.season_start
        WHERE pq.season_start = :season_start
          AND psa.vote_avg IS NOT NULL
          AND pim.player_fotmob_id IS NOT NULL
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(primary_sql, {"season_start": season_start}).fetchall()
        if not rows:
            log.info(
                "Primary MANTRA actuals query returned 0 rows for season %s; trying fallback",
                season_start,
            )
            rows = conn.execute(fallback_sql, {"season_start": season_start}).fetchall()

    actuals = {int(r[0]): float(r[1]) for r in rows}
    if not actuals:
        log.warning("No actual ratings found for season %d", season_start)
        return None

    # Compare against ``FP_Mantra`` (0-100 "merit score" pillar aggregate),
    # NOT ``VR`` ("Valore Reale"). VR is an auction-value index (clip
    # 0-300, centred ~100) that encodes convenience-relative-to-price, not
    # an estimate of the player's rating.
    y_true: list[float] = []
    y_pred: list[float] = []
    for p in players:
        fid = p.get("fantacalcio_id")
        fp_mantra = p.get("FP_Mantra")
        if fid is None or fp_mantra is None:
            continue
        try:
            fid_int = int(fid)
        except (TypeError, ValueError):
            continue
        if fid_int in actuals:
            y_true.append(actuals[fid_int])
            y_pred.append(float(fp_mantra))

    if len(y_true) < 10:
        log.warning("Too few matched players (%d) for MANTRA evaluation", len(y_true))
        return None

    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    n = len(y_true_arr)

    if np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        log.warning(
            "MANTRA evaluation: degenerate distribution (zero variance in "
            "actuals or predictions) — rank correlation is undefined"
        )
        return None

    from scipy.stats import spearmanr

    spearman_corr, spearman_p = spearmanr(y_pred_arr, y_true_arr)

    # Top-20% precision: of the players actually best-rated (top 20% by
    # vote_avg), what fraction does MANTRA also rank in its own top 20%
    # (by FP_Mantra)? Directly meaningful for MANTRA's real use case
    # (auction/draft prioritisation) and unaffected by scale differences.
    k = max(1, round(n * 0.20))
    true_top_idx = set(np.argsort(-y_true_arr)[:k])
    pred_top_idx = set(np.argsort(-y_pred_arr)[:k])
    top20_precision = len(true_top_idx & pred_top_idx) / k

    metrics = {
        "fp_mantra_vote_spearman": round(float(spearman_corr), 4),
        "fp_mantra_vote_top20_precision": round(float(top20_precision), 4),
    }
    log.info(
        "MANTRA evaluation: Spearman ρ=%.4f (p=%.4g), top20 precision=%.4f (n=%d)",
        spearman_corr,
        spearman_p,
        top20_precision,
        n,
    )

    run_id = f"mantra-{season_start}-{uuid.uuid4().hex[:8]}"
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    """
                    INSERT INTO model_runs
                        (run_id, model_name, trained_at, season_start, git_commit, status)
                    VALUES
                        (:run_id, 'mantra-4pillar', NOW(), :season_start, :git_commit, 'completed')
                    ON CONFLICT (run_id) DO NOTHING
                    """
                ),
                {
                    "run_id": run_id,
                    "season_start": season_start,
                    "git_commit": _git_commit(),
                },
            )

            for metric_name, value in metrics.items():
                conn.execute(
                    sa.text(
                        """
                        INSERT INTO model_metrics
                            (run_id, metric_name, metric_value, split)
                        VALUES (:run_id, :metric_name, :metric_value, 'test')
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "run_id": run_id,
                        "metric_name": metric_name,
                        "metric_value": value,
                    },
                )

        log.info("MANTRA metrics persisted as run %s", run_id)
    except Exception as exc:
        log.error("Failed to persist MANTRA metrics (non-fatal): %s", exc)

    return metrics