from __future__ import annotations

"""Target variable: fantavoto_medio (average fantasy-football rating).

Two modes:
1. **External CSV** — supply a CSV with columns
       [player_fotmob_id, season_start, fantavoto_medio]
   or   [player_name, season_label, fantavoto_medio]
   This is the preferred mode when actual Fantacalcio data is available.

2. **Approximation** — compute from FotMob stats using a role-aware
   Fantacalcio bonus/malus formula.  Different roles receive different
   weights for the same event (e.g. a DEF goal scores higher than a FWD
   goal; a GK clean sheet yields a larger bonus than a DEF clean sheet).

Assumptions:
- Column names in *df* have already been canonicalised by
  ``ml.data.stat_names.canonicalize_columns`` (applied in loader.py).
- "appearances" or "mins_played" must be present to compute per-match stats.
- FotMob ``rating`` is intentionally excluded from this formula: it is a
  holistic per-match score that already encodes goals/assists/defensive
  actions — including it would create circular bias for any model trained
  on event-level stats.
- All formula terms are (season_total / matches_played) to match
  Fantacalcio's per-game scoring convention.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Role-based weight configuration ──────────────────────────────────────────
# Keys must match canonical column names produced by stat_names.canonicalize_columns().
# Rationale for weight choices:
#   - DEF/GK goals are rarer and worth more in Fantacalcio classic rules.
#   - clean_sheet is a per-match rate (0–1); GK weight higher than DEF.
#   - saves: small per-stop bonus; _goals_prevented captures xG quality.
#   - big_chance_missed: mild penalty for FWD to reward efficiency.
#   - Defensive counting stats (tackle, intercept, clearance) are included
#     only at small weights to break ties, not dominate the formula.

_BASE_RATING: float = 6.0

WEIGHTS_BY_ROLE: dict[str, dict[str, float]] = {
    "GK": {
        "goals":            4.5,
        "goal_assist":      3.0,
        "clean_sheet":      1.0,   # per-match rate × weight
        "saves":            0.03,  # per save
        "_goals_prevented": 0.5,   # per xG-unit prevented
        "yellow_card":     -0.5,
        "yellow_red_card": -1.0,
        "red_card":        -1.0,
        "own_goals":       -2.0,
        "penalty_scored":   3.0,
        "penalty_missed":  -3.0,
    },
    "DEF": {
        "goals":               4.0,
        "goal_assist":         2.5,
        "clean_sheet":         1.5,
        "total_tackle":        0.04,
        "interception":        0.04,
        "effective_clearance": 0.02,
        "penalty_won":         0.5,
        "penalty_conceded":   -0.5,
        "yellow_card":        -0.5,
        "yellow_red_card":    -1.0,
        "red_card":           -1.0,
        "own_goals":          -2.0,
        "penalty_scored":      3.0,
        "penalty_missed":     -3.0,
    },
    "MID": {
        "goals":             3.5,
        "goal_assist":       1.5,
        "big_chance_created": 0.1,
        "total_tackle":      0.02,
        "interception":      0.02,
        "penalty_won":       0.5,
        "penalty_conceded": -0.3,
        "yellow_card":      -0.5,
        "yellow_red_card":  -1.0,
        "red_card":         -1.0,
        "own_goals":        -2.0,
        "penalty_scored":    3.0,
        "penalty_missed":   -3.0,
    },
    "FWD": {
        "goals":            3.0,
        "goal_assist":      1.0,
        "big_chance_missed": -0.15,
        "penalty_won":      0.5,
        "yellow_card":     -0.5,
        "yellow_red_card": -1.0,
        "red_card":        -1.0,
        "own_goals":       -2.0,
        "penalty_scored":   3.0,
        "penalty_missed":  -3.0,
    },
}


def _safe_col(df: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
    """Return the column series if it exists, else None."""
    if col_name in df.columns:
        return df[col_name]
    log.debug("stat column not found in DataFrame: %r", col_name)
    return None


def _estimate_matches(df: pd.DataFrame) -> pd.Series:
    """Best estimate of matches played per player-season.

    Expects canonical column names (post-canonicalize_columns).
    """
    apps = _safe_col(df, "appearances")
    mins = _safe_col(df, "mins_played")

    if apps is not None:
        raw = pd.to_numeric(apps, errors="coerce")
    elif mins is not None:
        raw = pd.to_numeric(mins, errors="coerce") / 90.0
    else:
        raise ValueError(
            "Neither 'appearances' nor 'mins_played' found in player stats. "
            "Cannot compute per-match ratings."
        )
    return raw.clip(lower=1).fillna(1)


def _per_match_series(
    df: pd.DataFrame, col_name: str, matches: pd.Series
) -> pd.Series:
    """Return col / matches, or all-zeros if col is absent."""
    col = _safe_col(df, col_name)
    if col is None:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(col, errors="coerce").fillna(0.0) / matches


def _compute_role_contribution(
    df: pd.DataFrame,
    role: str,
    matches: pd.Series,
) -> pd.Series:
    """Accumulate weighted per-match stat contributions for a given role.

    Uses WEIGHTS_BY_ROLE[role]; missing stats silently contribute 0.0.
    """
    weights = WEIGHTS_BY_ROLE.get(role, WEIGHTS_BY_ROLE["FWD"])
    contribution = pd.Series(0.0, index=df.index)
    for stat, weight in weights.items():
        contribution += _per_match_series(df, stat, matches) * weight
    return contribution


def compute_approx_fantavoto(df: pd.DataFrame) -> pd.Series:
    """Return a Series (fantavoto_medio) estimated from FotMob stats.

    Applies role-specific weights from WEIGHTS_BY_ROLE.  Requires the
    DataFrame to have a 'canonical_role' column ('GK'|'DEF'|'MID'|'FWD');
    rows with missing role default to 'FWD'.

    All stat columns must use canonical names (see stat_names.py).
    The result is clipped to [1.0, 10.0].
    """
    matches = _estimate_matches(df)

    role_col: pd.Series
    if "canonical_role" in df.columns:
        role_col = df["canonical_role"].fillna("FWD").astype(str)
    else:
        log.warning(
            "No 'canonical_role' column found; applying FWD weights to all rows. "
            "Run the scraper role-fetch step and update loader.py."
        )
        role_col = pd.Series("FWD", index=df.index)

    rating = pd.Series(_BASE_RATING, index=df.index, dtype=float)

    for role in WEIGHTS_BY_ROLE:
        mask = role_col == role
        if not mask.any():
            continue
        rating.loc[mask] += _compute_role_contribution(
            df.loc[mask], role, matches.loc[mask]
        )

    # Rows with an unrecognised role fall through the loop with 0 contribution.
    unassigned = ~role_col.isin(WEIGHTS_BY_ROLE)
    if unassigned.any():
        log.warning(
            "%d rows have unrecognized canonical_role values: %s — applying FWD weights.",
            unassigned.sum(),
            role_col[unassigned].unique().tolist(),
        )
        rating.loc[unassigned] += _compute_role_contribution(
            df.loc[unassigned], "FWD", matches.loc[unassigned]
        )

    return rating.clip(1.0, 10.0)


def attach_target(
    df: pd.DataFrame,
    external_csv: Optional[Path] = None,
    min_minutes: int = 800,
) -> pd.DataFrame:
    """Add ``fantavoto_medio`` column to *df* and drop under-represented players.

    Args:
        df: Wide-format player-season DataFrame from :func:`~data.loader.load_raw_data`.
        external_csv: Optional path to a CSV with columns:
            ``player_fotmob_id, season_start, fantavoto_medio``
            OR ``player_name, season_label, fantavoto_medio``.
        min_minutes: Rows whose ``mins_played`` is below this threshold
            are dropped (noisy target).

    Returns:
        DataFrame with a ``fantavoto_medio`` column; rows with NaN targets removed.
    """
    if external_csv is not None:
        log.info("Loading external fantavoto data from %s", external_csv)
        ext = pd.read_csv(external_csv)

        if "player_fotmob_id" in ext.columns and "season_start" in ext.columns:
            df = df.merge(
                ext[["player_fotmob_id", "season_start", "fantavoto_medio"]],
                on=["player_fotmob_id", "season_start"],
                how="left",
            )
        elif "player_name" in ext.columns and "season_label" in ext.columns:
            df = df.merge(
                ext[["player_name", "season_label", "fantavoto_medio"]],
                on=["player_name", "season_label"],
                how="left",
            )
        else:
            raise ValueError(
                "External CSV must contain either "
                "(player_fotmob_id, season_start) or (player_name, season_label)."
            )

        missing = df["fantavoto_medio"].isna().sum()
        if missing:
            log.warning(
                "%d player-seasons have no external fantavoto value; "
                "computing approximation for them.",
                missing,
            )
            mask = df["fantavoto_medio"].isna()
            df.loc[mask, "fantavoto_medio"] = compute_approx_fantavoto(
                df.loc[mask]
            )
    else:
        log.info(
            "No external fantavoto CSV supplied — computing approximation from stats."
        )
        df["fantavoto_medio"] = compute_approx_fantavoto(df)

    # ── Quality filter ────────────────────────────────────────────────────────
    mins_col = _safe_col(df, "mins_played")
    if mins_col is None:
        mins_col = _safe_col(df, "minutesPlayed")
    if mins_col is not None:
        low_sample = pd.to_numeric(mins_col, errors="coerce").fillna(0) < min_minutes
        dropped = int(low_sample.sum())
        if dropped:
            log.info(
                "Dropping %d player-seasons with fewer than %d minutes played.",
                dropped,
                min_minutes,
            )
        df = df[~low_sample].reset_index(drop=True)
    else:
        log.warning("'mins_played' not found; skipping min_minutes filter.")

    before = len(df)
    df = df.dropna(subset=["fantavoto_medio"])
    after = len(df)
    if before - after:
        log.warning("Dropped %d rows with NaN fantavoto_medio.", before - after)

    log.info("Dataset after target attachment: %d rows", len(df))
    return df
