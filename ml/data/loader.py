from __future__ import annotations

"""Data loading from the FotMob PostgreSQL platform.

Responsibilities:
- Pull ``player_season_stats`` (long format) and pivot to one row per
  (player, season) with stat categories as columns.
- Pull ``team_season_stats`` and derive team-strength features.
- Merge both datasets.
- Apply quality filters (minimum matches, league scope).

Assumptions:
- FotMob stat category slugs are the raw strings stored in ``stat_category``
  (e.g. "goals", "goalAssist", "yellowCards", "minutesPlayed", "appearances").
- One player can appear in multiple teams per season; we keep the row with
  the highest minutes (or first if equal) to represent the dominant team.
"""

import logging
from typing import Optional

import pandas as pd
import sqlalchemy as sa

from ..config import MLConfig
from .stat_names import canonicalize_columns

log = logging.getLogger(__name__)

# ── SQL templates ─────────────────────────────────────────────────────────────

_PLAYER_STATS_SQL = """
SELECT
    pss.player_fotmob_id,
    pss.player_name,
    pss.team_fotmob_id,
    pss.team_name,
    pss.stat_category,
    pss.value,
    pss.rank                AS stat_rank,
    s.season_start,
    s.season_label,
    l.name                  AS league_name
FROM player_season_stats pss
JOIN seasons  s ON s.id = pss.season_id
JOIN leagues  l ON l.id = s.league_id
{where_clause}
ORDER BY s.season_start, pss.player_fotmob_id, pss.stat_category
"""

_TEAM_STATS_SQL = """
SELECT
    tss.team_fotmob_id,
    tss.team_name,
    tss.stat_category,
    tss.value,
    tss.rank                AS team_rank,
    s.season_start,
    s.season_label,
    l.name                  AS league_name
FROM team_season_stats tss
JOIN seasons  s ON s.id = tss.season_id
JOIN leagues  l ON l.id = s.league_id
{where_clause}
ORDER BY s.season_start, tss.team_fotmob_id, tss.stat_category
"""

_PLAYER_PROFILES_SQL = """
SELECT player_fotmob_id, canonical_role
FROM player_profiles
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_where(league_name: Optional[str]) -> str:
    if league_name:
        escaped = league_name.replace("'", "''")
        return f"WHERE l.name ILIKE '%{escaped}%'"
    return ""


def _pivot_stats(df_long: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Pivot stat_category rows into wide-format columns."""
    df_wide = df_long.pivot_table(
        index=index_cols,
        columns="stat_category",
        values="value",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None
    return df_wide


def _deduplicate_multi_team_players(df: pd.DataFrame) -> pd.DataFrame:
    """When a player appears for >1 team in the same season, keep the row
    for the team with the most minutes (proxy for dominant spell)."""
    minutes_col = next(
        (c for c in df.columns if "minute" in c.lower()), None
    )
    if minutes_col is None:
        # No minutes column: just keep the first occurrence
        return df.drop_duplicates(
            subset=["player_fotmob_id", "season_start"], keep="first"
        )

    df = df.sort_values(
        ["player_fotmob_id", "season_start", minutes_col],
        ascending=[True, True, False],
        na_position="last",
    )
    return df.drop_duplicates(
        subset=["player_fotmob_id", "season_start"], keep="first"
    )


# ── Team-strength features ─────────────────────────────────────────────────────

# FotMob team stat categories used for strength scoring (use what's available).
# Keys must match the canonical snake_case names stored in team_season_stats.
_TEAM_STRENGTH_CATS = {
    "rating_team":      1.0,  # overall FotMob team rating (best proxy for wins)
    "goals_team_match": 0.5,  # goals scored
    "clean_sheet_team": 0.3,  # clean sheets
}


def _build_team_strength(df_team_long: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns:
        team_fotmob_id, season_start, team_strength_score, team_rank_norm
    """
    df_wide = df_team_long.pivot_table(
        index=["team_fotmob_id", "team_name", "season_start"],
        columns="stat_category",
        values="value",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None

    # Weighted sum of available strength proxies
    score = pd.Series(0.0, index=df_wide.index)
    for cat, weight in _TEAM_STRENGTH_CATS.items():
        if cat in df_wide.columns:
            col = df_wide[cat].fillna(0)
            # Normalise within season
            season_max = df_wide.groupby("season_start")[cat].transform("max").replace(0, 1)
            score += (col / season_max) * weight

    df_wide["team_strength_score"] = score

    # is_top_team: top-3 teams by strength score each season
    df_wide["is_top_team"] = (
        df_wide.groupby("season_start")["team_strength_score"]
        .rank(method="min", ascending=False)
        <= 3
    ).astype(int)

    # Normalised team rank from "wins" if available, else strength score
    rank_source = "wins" if "wins" in df_wide.columns else "team_strength_score"
    df_wide["team_rank_norm"] = (
        df_wide.groupby("season_start")[rank_source]
        .rank(method="min", ascending=False, na_option="bottom")
        .div(df_wide.groupby("season_start")[rank_source].transform("count"))
    )

    keep = [
        "team_fotmob_id", "season_start",
        "team_strength_score", "is_top_team", "team_rank_norm",
    ]
    return df_wide[[c for c in keep if c in df_wide.columns]].copy()


# ── Public interface ──────────────────────────────────────────────────────────

def load_raw_data(engine: sa.Engine, cfg: MLConfig) -> pd.DataFrame:
    """Load and merge player + team stats. Returns the feature DataFrame.

    Columns after this step:
    - player_fotmob_id, player_name, team_fotmob_id, team_name
    - season_start, season_label, league_name
    - canonical_role ('GK'|'DEF'|'MID'|'FWD'; 'FWD' when unknown)
    - <stat_category_*> columns (canonical snake_case names, one per FotMob stat)
    - team_strength_score, is_top_team, team_rank_norm
    """
    where = _build_where(cfg.league_name)

    log.info("Loading player_season_stats …")
    df_player_long = pd.read_sql(
        sa.text(_PLAYER_STATS_SQL.format(where_clause=where)),
        engine,
    )
    if df_player_long.empty:
        raise ValueError(
            "No player_season_stats rows found. Have you run the scraper?"
        )
    log.info("  %d long-format rows for %d distinct players across %d seasons",
             len(df_player_long),
             df_player_long["player_fotmob_id"].nunique(),
             df_player_long["season_start"].nunique())

    index_cols = [
        "player_fotmob_id", "player_name",
        "team_fotmob_id", "team_name",
        "season_start", "season_label", "league_name",
    ]
    df_player = _pivot_stats(df_player_long, index_cols)
    df_player = canonicalize_columns(df_player)
    df_player = _deduplicate_multi_team_players(df_player)

    # ── Attach player role ────────────────────────────────────────────────────
    try:
        df_profiles = pd.read_sql(sa.text(_PLAYER_PROFILES_SQL), engine)
        if not df_profiles.empty:
            df_player = df_player.merge(
                df_profiles[["player_fotmob_id", "canonical_role"]],
                on="player_fotmob_id",
                how="left",
            )
            df_player["canonical_role"] = (
                df_player["canonical_role"].fillna("FWD").astype(str)
            )
            log.info(
                "Role distribution: %s",
                df_player["canonical_role"].value_counts().to_dict(),
            )
        else:
            log.warning(
                "player_profiles table is empty — run the scraper role-fetch step. "
                "Defaulting all roles to 'FWD'."
            )
            df_player["canonical_role"] = "FWD"
    except Exception:
        log.warning(
            "Could not load player_profiles (table may not exist yet). "
            "Defaulting all roles to 'FWD'. Run db migration 001_add_player_profiles.sql.",
            exc_info=True,
        )
        df_player["canonical_role"] = "FWD"

    log.info("Loading team_season_stats …")
    df_team_long = pd.read_sql(
        sa.text(_TEAM_STATS_SQL.format(where_clause=where)),
        engine,
    )
    if not df_team_long.empty:
        df_team_strength = _build_team_strength(df_team_long)
        df_player = df_player.merge(
            df_team_strength,
            on=["team_fotmob_id", "season_start"],
            how="left",
        )
        log.info("  Team strength features merged.")
    else:
        log.warning("No team_season_stats found; skipping team strength features.")

    log.info("Raw dataset shape: %s", df_player.shape)
    return df_player
