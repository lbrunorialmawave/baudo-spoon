"""Schedule-Adjusted Performance (SAP) features.

SAP adjusts per-90 stats by opponent schedule strength:

    sap_stat = raw_stat_per90 * sap_weight

where:

    opponent_mean_rank = (league_total_rank − own_rank) / (n_teams − 1)
    sap_weight = opponent_mean_rank / league_mean_rank

A ``sap_weight > 1.0`` means the player faced stronger-than-average opponents,
so their raw stat is weighted UPWARD — a strong performance against strong
opponents is worth more than the same number against weak opponents.

This direction is intentional and must not be inverted.
See ``ScheduleAdjustmentConfig`` for the normalised coefficient range.

Note on ``MissingDataPolicy``:
    We intentionally use ``IMPUTE_ZERO`` (not ``PROXY_FEATURE``) and declare
    only ``stat_col`` as a required column.  The schedule columns
    (``team_rank_norm``, ``season_start``, ``league_name``) are checked
    *inside* ``compute()`` with a graceful fallback to ``sap_weight=1.0``
    (no adjustment) when absent.  This avoids the ``PROXY_FEATURE`` policy's
    ``__init_subclass__`` requirement for a class-level ``proxy_feature_name``
    string, which can't be expressed generically for per-stat subclasses.
"""

from __future__ import annotations

import logging

import polars as pl

from ml.domain.features import Feature, MissingDataPolicy

__all__ = ["ALL_SAP_FEATURES", "SapFeature"]

log = logging.getLogger(__name__)

_SCHEDULE_COLS = frozenset(["team_rank_norm", "season_start", "league_name"])


class SapFeature(Feature):
    """SAP-adjusted version of a per-90 stat.

    Required column: the source per-90 column (``stat_col``).
    Optional schedule columns: ``team_rank_norm``, ``season_start``,
    ``league_name``, ``team_fotmob_id``.  When absent, ``sap_weight=1.0``
    (no adjustment) is applied silently.
    """

    stat_col: str
    # ponytail: only stat_col is truly required; schedule cols handled in compute()
    missing_data_policy = MissingDataPolicy.IMPUTE_ZERO

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.stat_col}_sap"

    @property
    def required_columns(self) -> frozenset[str]:  # type: ignore[override]
        return frozenset([self.stat_col])

    def compute(self, data: pl.DataFrame) -> pl.Series:
        raw = data[self.stat_col].cast(pl.Float64).fill_null(0.0)

        # If schedule info is absent, return raw stat unmodified (sap_weight=1.0).
        if not _SCHEDULE_COLS.issubset(data.columns):
            missing = sorted(_SCHEDULE_COLS - frozenset(data.columns))
            log.info(
                "SapFeature '%s': schedule columns %s absent; sap_weight=1.0.",
                self.name,
                missing,
            )
            return raw

        # Deduplicate to one row per (team, season, league) before aggregating so
        # that teams with many players don't inflate the league totals.
        group_cols = ["season_start", "league_name"]
        if "team_fotmob_id" in data.columns:
            dedup = data.unique(subset=group_cols + ["team_fotmob_id"])
        else:
            dedup = data

        league_stats = dedup.group_by(group_cols).agg(
            [
                pl.col("team_rank_norm").sum().alias("_league_sum"),
                pl.col("team_rank_norm").len().alias("_n_teams"),
                pl.col("team_rank_norm").mean().alias("_league_mean"),
            ]
        )

        df = data.join(league_stats, on=group_cols, how="left")

        global_mean = float(data["team_rank_norm"].mean() or 1.0)
        df = df.with_columns(
            [
                pl.col("_league_sum").fill_null(global_mean),
                pl.col("_n_teams").fill_null(10),
                pl.col("_league_mean").fill_null(global_mean),
            ]
        )

        own_rank = df["team_rank_norm"].fill_null(global_mean)
        n_minus_1 = (df["_n_teams"].cast(pl.Float64) - 1.0).clip(lower_bound=1.0)
        opp_mean = (df["_league_sum"].cast(pl.Float64) - own_rank) / n_minus_1
        league_mean = df["_league_mean"].cast(pl.Float64).clip(lower_bound=1e-6)

        sap_weight = (opp_mean / league_mean).clip(lower_bound=0.1, upper_bound=10.0)
        return (raw * sap_weight).alias(self.name)


# ── Concrete instances ────────────────────────────────────────────────────────

_SAP_STAT_COLS: list[str] = [
    "goals_per90",
    "goal_assist_per90",
    "total_scoring_att_per90",
    "ontarget_scoring_att_per90",
    "saves_per90",
    "_goals_prevented_per90",
]

ALL_SAP_FEATURES: list[SapFeature] = [
    type(f"Sap_{col}", (SapFeature,), {"stat_col": col})() for col in _SAP_STAT_COLS
]
