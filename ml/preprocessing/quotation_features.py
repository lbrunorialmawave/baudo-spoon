"""Feature engineering from Fantacalcio market quotations.

Reads ``player_quotations`` and ``player_id_map`` from the database and
adds market-signal features to the wide-format player-season DataFrame
produced by :mod:`ml.data.loader`.

Features produced (in priority order):

1. ``qt_a_norm``            — current valuation / 300 (cross-season
                              comparable share of the 300-credit budget).
                              This is the most direct market signal and
                              has the highest expected importance.
2. ``price_delta_pct``      — (qt_a − qt_i) / qt_i, the per-season
                              "value momentum". Positive means the
                              player outperformed the auction consensus.
3. ``qt_a_vs_role_median``  — z-score of qt_a within the same
                              (season_start, role) cohort. Captures
                              whether the player is in the top / mid /
                              bottom tier of his positional market.
4. ``price_trend_2y``       — relative change of qt_a vs. the prior
                              season's qt_a. Requires the player to be
                              present in two consecutive seasons.
5. ``qt_a_lag1`` / ``qt_i_lag1`` — raw lagged values, useful for
                              sequence models and the delta feature.

All features are nullable: a player without a quotation row simply gets
``NaN`` and the model treats it as "missing" via the existing
``impute_environmental_features`` path.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa

log = logging.getLogger(__name__)

#: Number of credits that define the default Fantacalcio budget.
DEFAULT_BUDGET = 300

#: Quotations are joined on (player_fotmob_id, season_start). The
#: auxiliary tables (player_id_map, player_quotations) are read with
#: this single query and then merged into the player DataFrame.
_JOINED_QUOTATIONS_SQL = sa.text("""
    SELECT
        pim.player_fotmob_id,
        pq.season_start,
        pq.role,
        pq.team,
        pq.qt_a,
        pq.qt_i,
        pq.diff_val,
        pq.qt_a_norm,
        pq.qt_i_norm,
        pq.fvm,
        pim.match_method,
        pim.confidence
    FROM player_quotations pq
    JOIN player_id_map     pim
      ON pim.fantacalcio_id = pq.fantacalcio_id
     AND pim.season_start   = pq.season_start
    WHERE pim.player_fotmob_id IS NOT NULL
""")


def _load_quotation_frame(engine: sa.Engine) -> pd.DataFrame:
    """Fetch the joined (player_fotmob_id, season_start) quotation view."""
    return pd.read_sql(_JOINED_QUOTATIONS_SQL, engine)


def attach_quotation_features(
    df: pd.DataFrame,
    engine: sa.Engine,
    budget: int = DEFAULT_BUDGET,
) -> pd.DataFrame:
    """Attach all quotation-based features to *df* in-place.

    Returns the same DataFrame for chaining convenience.
    """
    if not {"player_fotmob_id", "season_start"}.issubset(df.columns):
        raise ValueError(
            "df must contain 'player_fotmob_id' and 'season_start' columns."
        )

    quotes = _load_quotation_frame(engine)
    if quotes.empty:
        log.warning(
            "No quotations found in player_quotations / player_id_map. "
            "Quotation features will be NaN. "
            "Run `python -m ml.data.import_quotations` first."
        )
        for col in (
            "qt_a", "qt_i", "qt_a_norm", "qt_i_norm",
            "price_delta_pct", "qt_a_vs_role_median",
            "price_trend_2y", "qt_a_lag1",
        ):
            df[col] = np.nan
        return df

    quotes = quotes.drop_duplicates(
        subset=["player_fotmob_id", "season_start"]
    )
    # Rename role so the merge target is unambiguous.
    quotes = quotes.rename(columns={"role": "quotation_role"})

    # ── Left-join quotation snapshot onto the main DataFrame ─────────────
    df = df.merge(
        quotes[
            [
                "player_fotmob_id", "season_start",
                "qt_a", "qt_i", "qt_a_norm", "qt_i_norm",
                "diff_val", "fvm", "quotation_role", "match_method",
            ]
        ],
        on=["player_fotmob_id", "season_start"],
        how="left",
    )

    # ── 1. qt_a_norm (priority 1 — already produced by the SQL column) ────
    # The column is already in the merge, but ensure dtype is float.
    df["qt_a_norm"] = pd.to_numeric(df["qt_a_norm"], errors="coerce")
    df["qt_i_norm"] = pd.to_numeric(df["qt_i_norm"], errors="coerce")
    # Backfill: if the SQL generated column is missing (e.g. pre-migration
    # DB), compute it from raw qt_a / qt_i with the configured budget.
    mask = df["qt_a_norm"].isna() & df["qt_a"].notna()
    if mask.any():
        df.loc[mask, "qt_a_norm"] = df.loc[mask, "qt_a"] / float(budget)
    mask_i = df["qt_i_norm"].isna() & df["qt_i"].notna()
    if mask_i.any():
        df.loc[mask_i, "qt_i_norm"] = df.loc[mask_i, "qt_i"] / float(budget)

    # Cast to float once so all subsequent arithmetic / comparisons are
    # safe from the Int64 / boolean NA ambiguity in numpy.
    qt_a = pd.to_numeric(df["qt_a"], errors="coerce").astype(float)
    qt_i = pd.to_numeric(df["qt_i"], errors="coerce").astype(float)

    # ── 2. price_delta_pct: (qt_a − qt_i) / qt_i (priority 2) ─────────────
    # 0% = consensus hit; +100% = doubled; -50% = halved. NaN if either
    # side is missing or qt_i is zero.
    valid_i = qt_i.gt(0) & qt_a.notna() & qt_i.notna()
    df["price_delta_pct"] = np.where(
        valid_i.fillna(False),
        (qt_a - qt_i) / qt_i,
        np.nan,
    )
    # Clip to a sensible range to avoid one-record outliers dominating
    # the model. Real-world values rarely exceed [-0.8, +2.0].
    df["price_delta_pct"] = df["price_delta_pct"].clip(-0.8, 2.0)

    # ── 3. qt_a_vs_role_median: z-score within (season, role) (priority 3)
    grouped = df.groupby(["season_start", "canonical_role"])[qt_a.name]
    role_median = grouped.transform("median")
    role_std = grouped.transform("std")
    valid_std = role_std.gt(0) & qt_a.notna()
    df["qt_a_vs_role_median"] = np.where(
        valid_std.fillna(False),
        (qt_a - role_median) / role_std,
        np.nan,
    )

    # ── 4 & 5. Lagged qt_a / qt_i + price_trend_2y (priority 4) ───────────
    # Must be sorted and grouped by player to keep the temporal order.
    df = df.sort_values(["player_fotmob_id", "season_start"])
    grp = df.groupby("player_fotmob_id", sort=False)
    df["qt_a_lag1"] = pd.to_numeric(
        grp["qt_a"].shift(1), errors="coerce"
    ).astype(float)
    df["qt_i_lag1"] = pd.to_numeric(
        grp["qt_i"].shift(1), errors="coerce"
    ).astype(float)
    # price_trend_2y: relative change vs. previous season.
    valid_lag = df["qt_a_lag1"].gt(0) & qt_a.notna()
    df["price_trend_2y"] = np.where(
        valid_lag.fillna(False),
        (qt_a - df["qt_a_lag1"]) / df["qt_a_lag1"],
        np.nan,
    )
    df["price_trend_2y"] = df["price_trend_2y"].clip(-0.8, 2.0)

    # Restore the original row order.
    df = df.sort_index()

    # ── Diagnostics ────────────────────────────────────────────────────────
    n_with = int(df["qt_a_norm"].notna().sum())
    n_total = len(df)
    pct = (100.0 * n_with / n_total) if n_total else 0.0
    log.info(
        "Quotation features: %d / %d rows have a matched quotation (%.1f%%).",
        n_with, n_total, pct,
    )
    log.info(
        "  match_method distribution (among non-null): %s",
        df.loc[df["qt_a"].notna(), "match_method"].value_counts().to_dict()
        if "match_method" in df.columns else "n/a",
    )

    return df


# ── Convenience for ad-hoc inspection ───────────────────────────────────────

def quote_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season coverage summary. Used by the import CLI for QA logs."""
    if "qt_a_norm" not in df.columns:
        raise ValueError("Run attach_quotation_features first.")
    return (
        df.groupby("season_start")
        .agg(
            n_total=("player_fotmob_id", "count"),
            n_with_quote=("qt_a_norm", "count"),
            median_qt_a=("qt_a", "median"),
        )
        .assign(
            coverage=lambda t: (t["n_with_quote"] / t["n_total"]).round(3),
        )
        .reset_index()
    )
