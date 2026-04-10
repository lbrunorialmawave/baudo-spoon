from __future__ import annotations

"""Canonical stat name mapping for FotMob statistics.

FotMob exposes stat slugs in two formats depending on the endpoint:
- camelCase (e.g. "goalAssist", "yellowCards") — legacy JS scraper
- snake_case (e.g. "goal_assist", "yellow_card") — deep stats API

This module is the single source of truth. Apply ``canonicalize_columns()``
immediately after pivot so all downstream code (target.py, features.py)
sees stable column names regardless of which FotMob endpoint was scraped.
"""

import pandas as pd

# Maps every known FotMob variant → canonical snake_case name.
# Rules:
# - plural forms collapse to singular where Fantacalcio counts per-event
# - camelCase variants are added alongside snake_case
# - GK stats with leading underscore are preserved as-is (FotMob convention)
CANONICAL_STAT_NAMES: dict[str, str] = {
    # ── Appearances / playing time ────────────────────────────────────────────
    "minutesPlayed": "mins_played",
    "minutesPlayed90s": "mins_played",
    "minutes_played": "mins_played",
    "matchesPlayed": "appearances",
    "matches_played": "appearances",
    # ── Goals ─────────────────────────────────────────────────────────────────
    "goals": "goals",
    # ── Assists ───────────────────────────────────────────────────────────────
    "goalAssist": "goal_assist",
    "goal_assists": "goal_assist",
    # ── Expected stats ────────────────────────────────────────────────────────
    "expectedGoals": "expected_goals",
    "expected_goals_per90": "expected_goals_per_90",
    "expectedAssists": "expected_assists",
    "expected_assists_per90": "expected_assists_per_90",
    "expectedGoalsOnTarget": "expected_goalsontarget",
    # ── Shots ─────────────────────────────────────────────────────────────────
    "totalScoringAtt": "total_scoring_att",
    "total_scoring_attempts": "total_scoring_att",
    "ontargetScoringAtt": "ontarget_scoring_att",
    "onTargetScoringAtt": "ontarget_scoring_att",
    # ── Disciplinary ──────────────────────────────────────────────────────────
    "yellowCards": "yellow_card",
    "yellow_cards": "yellow_card",
    "yellowRedCards": "yellow_red_card",
    "yellow_red_cards": "yellow_red_card",
    "redCards": "red_card",
    "red_cards": "red_card",
    # ── Penalties ─────────────────────────────────────────────────────────────
    "penaltiesScored": "penalty_scored",
    "penalties_scored": "penalty_scored",
    "penaltiesMissed": "penalty_missed",
    "penalties_missed": "penalty_missed",
    "penaltiesWon": "penalty_won",
    "penalties_won": "penalty_won",
    "penaltyConceded": "penalty_conceded",
    "penalties_conceded": "penalty_conceded",
    # ── Own goals ─────────────────────────────────────────────────────────────
    "ownGoals": "own_goals",
    # ── Passing ───────────────────────────────────────────────────────────────
    "accuratePass": "accurate_pass",
    "accurate_passes": "accurate_pass",
    "accurateLongBalls": "accurate_long_balls",
    "accurate_long_balls_won": "accurate_long_balls",
    # ── Chance creation ───────────────────────────────────────────────────────
    "keyPasses": "total_att_assist",
    "key_passes": "total_att_assist",
    "bigChancesCreated": "big_chance_created",
    "big_chances_created": "big_chance_created",
    "bigChancesMissed": "big_chance_missed",
    "big_chances_missed": "big_chance_missed",
    # ── Dribbles / duels ──────────────────────────────────────────────────────
    "successfulDribbles": "won_contest",
    "successful_dribbles": "won_contest",
    "duelsWon": "won_contest",
    # ── Defensive actions ─────────────────────────────────────────────────────
    "totalTackle": "total_tackle",
    "total_tackles": "total_tackle",
    "interceptions": "interception",
    "effectiveClearance": "effective_clearance",
    "effective_clearances": "effective_clearance",
    "outfielderBlock": "outfielder_block",
    "outfielder_blocks": "outfielder_block",
    "ballRecovery": "ball_recovery",
    "ball_recoveries": "ball_recovery",
    "defensiveContributions": "defensive_contributions",
    "possWonAtt3rd": "poss_won_att_3rd",
    "poss_won_def_3rd": "poss_won_att_3rd",
    # ── Fouls ─────────────────────────────────────────────────────────────────
    "foulsCommitted": "fouls",
    "fouls_committed": "fouls",
    # ── Goalkeeper ────────────────────────────────────────────────────────────
    "cleanSheet": "clean_sheet",
    "cleanSheets": "clean_sheet",
    "clean_sheets": "clean_sheet",
    "saves": "saves",
    "goalsConceded": "goals_conceded",
    "goals_conceded_total": "goals_conceded",
    # FotMob uses leading underscore for computed GK metrics — preserve as-is
    # "_save_percentage" → "_save_percentage" (identity, no rename needed)
    # "_goals_prevented" → "_goals_prevented" (identity, no rename needed)
}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns using CANONICAL_STAT_NAMES.

    Only columns present in the map are renamed; all others are untouched.
    When two source columns map to the same canonical name and both exist,
    the first one encountered (alphabetical sort of the rename map) wins;
    the duplicate is dropped to avoid ambiguous column names.

    Returns a shallow copy with renamed columns.
    """
    rename_map = {
        col: CANONICAL_STAT_NAMES[col]
        for col in df.columns
        if col in CANONICAL_STAT_NAMES
    }
    df = df.rename(columns=rename_map)

    # Drop duplicates that arose from multiple source columns mapping to same target.
    seen: set[str] = set()
    drop_cols: list[str] = []
    for col in df.columns:
        if col in seen:
            drop_cols.append(col)
        else:
            seen.add(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df
