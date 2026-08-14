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

import pandas as pd
import sqlalchemy as sa

from ..config import MLConfig
from .stat_names import canonicalize_columns

# Quotation-based features are an optional integration. The import is
# deferred to the call site because the module pulls a SQLAlchemy
# engine; loading it eagerly would couple the loader to the DB.
attach_quotation_features = None

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

# Season-scoped role lookup. Source of truth for ML feature pipelines —
# a row of `player_profiles` is the *current* role only, so historical
# training rows would silently inherit role changes that did not exist
# at the time of the season. This view gives us the role the player
# actually had per (player, season).
_PLAYER_SEASON_ROLES_SQL = """
SELECT player_fotmob_id, season_start, canonical_role
FROM player_season_roles
"""

# Cross-league fallback candidates (PR5 target-aware contract).
#
# Prefer player_stats_by_prediction_season (migration 026) filtered by the
# listino/target season so a historical backfill row for prediction=2024 is
# not silently replaced by a newer absolute-latest 2025 row.
# Falls back to player_latest_stats_any_league when the new view is absent
# (migration not yet applied) — see _append_foreign_fallback_rows.
_FOREIGN_FALLBACK_SQL_TARGET_AWARE = """
SELECT
    a.fantacalcio_id      AS player_fotmob_id,
    pq.player_name,
    pq.team                AS team_name,
    a.league_name,
    a.minutes_avg,
    a.goals_per90,
    a.assists_per90,
    a.saves_per90,
    a.clean_sheet_per90,
    a.source_season_start,
    a.prediction_season_start
FROM player_stats_by_prediction_season a
JOIN player_id_map pim
    ON pim.player_fotmob_id = a.fantacalcio_id
    AND pim.season_start = :season_start
JOIN player_quotations pq
    ON pq.fantacalcio_id = pim.fantacalcio_id
    AND pq.season_start = :season_start
WHERE a.prediction_season_start = :season_start
"""

# Legacy latest-absolute path (migration 018) — used only as fallback when
# player_stats_by_prediction_season is not available.
_FOREIGN_FALLBACK_SQL_LATEST = """
SELECT
    a.fantacalcio_id      AS player_fotmob_id,
    pq.player_name,
    pq.team                AS team_name,
    a.league_name,
    a.minutes_avg,
    a.goals_per90,
    a.assists_per90,
    a.saves_per90,
    a.clean_sheet_per90
FROM player_latest_stats_any_league a
JOIN player_id_map pim
    ON pim.player_fotmob_id = a.fantacalcio_id
    AND pim.season_start = :season_start
JOIN player_quotations pq
    ON pq.fantacalcio_id = pim.fantacalcio_id
    AND pq.season_start = :season_start
"""

# Back-compat alias for any external imports of the old name.
_FOREIGN_FALLBACK_SQL = _FOREIGN_FALLBACK_SQL_TARGET_AWARE

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_where(league_name: str | None) -> str:
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

def _attach_role(
    df_player: pd.DataFrame,
    engine: sa.Engine,
    log: logging.Logger,
) -> pd.DataFrame:
    """Attach a season-scoped role column to the player feature frame.

    Strategy:
    1. Try ``player_season_roles`` (preferred — season-aware).
    2. If the table does not exist, fall back to ``player_profiles``
       (current role, used as a constant for every season).
    3. If both fail, default every row to ``"FWD"`` so the pipeline
       still runs (matches pre-migration behaviour).
    """
    # 1. Season-aware source of truth
    try:
        df_season_roles = pd.read_sql(
            sa.text(_PLAYER_SEASON_ROLES_SQL), engine
        )
        if not df_season_roles.empty:
            df_player = df_player.merge(
                df_season_roles[
                    ["player_fotmob_id", "season_start", "canonical_role"]
                ],
                on=["player_fotmob_id", "season_start"],
                how="left",
            )
            missing_mask = df_player["canonical_role"].isna()
            n_defaulted = int(missing_mask.sum())
            df_player["canonical_role"] = (
                df_player["canonical_role"].fillna("FWD").astype(str)
            )
            log.info(
                "Role distribution (season-scoped): %s",
                df_player["canonical_role"].value_counts().to_dict(),
            )
            if n_defaulted > 0:
                _warn_high_cost_role_defaults(df_player, missing_mask, log)
            return df_player
        log.warning(
            "player_season_roles is empty — falling back to player_profiles. "
            "Re-run the scraper with --roles to populate it."
        )
    except sa.exc.ProgrammingError:
        log.warning(
            "player_season_roles table not found — falling back to "
            "player_profiles. Apply migration 003_add_player_season_roles.sql "
            "and re-run the scraper with --roles."
        )
    except Exception:
        log.warning(
            "Could not load player_season_roles; falling back to player_profiles.",
            exc_info=True,
        )

    # 2. Fallback: current role only
    try:
        df_profiles = pd.read_sql(sa.text(_PLAYER_PROFILES_SQL), engine)
        if not df_profiles.empty:
            df_player = df_player.merge(
                df_profiles[["player_fotmob_id", "canonical_role"]],
                on="player_fotmob_id",
                how="left",
            )
            missing_mask = df_player["canonical_role"].isna()
            n_defaulted = int(missing_mask.sum())
            df_player["canonical_role"] = (
                df_player["canonical_role"].fillna("FWD").astype(str)
            )
            log.info(
                "Role distribution (current-role fallback): %s",
                df_player["canonical_role"].value_counts().to_dict(),
            )
            if n_defaulted > 0:
                _warn_high_cost_role_defaults(df_player, missing_mask, log)
        else:
            log.warning(
                "player_profiles table is empty — defaulting all roles to 'FWD'."
            )
            df_player["canonical_role"] = "FWD"
    except Exception:
        log.warning(
            "Could not load player_profiles; defaulting all roles to 'FWD'.",
            exc_info=True,
        )
        df_player["canonical_role"] = "FWD"

    return df_player


def _warn_high_cost_role_defaults(
    df_player: pd.DataFrame,
    missing_mask: pd.Series,
    log: logging.Logger,
    *,
    high_cost_quantile: float = 0.5,
) -> None:
    """Emit an explicit WARNING for players defaulted to FWD that look expensive.

    Neo-arrivi without a resolved role silently become FWD, which can break
    formation constraints for GKs/DEFs that are priced like starters.
    """
    defaulted = df_player.loc[missing_mask]
    if defaulted.empty:
        return

    cost_col = "qt_a" if "qt_a" in defaulted.columns else None
    name_col = next(
        (c for c in ("player_name", "name", "name_fantacalcio", "name_fotmob")
         if c in defaulted.columns),
        None,
    )
    team_col = next(
        (c for c in ("team", "team_fantacalcio", "team_fotmob", "real_team")
         if c in defaulted.columns),
        None,
    )

    high_cost = defaulted
    if cost_col is not None and defaulted[cost_col].notna().any():
        threshold = float(defaulted[cost_col].median())
        if "qt_a" in df_player.columns and df_player["qt_a"].notna().any():
            threshold = max(
                threshold,
                float(df_player["qt_a"].quantile(high_cost_quantile)),
            )
        high_cost = defaulted[defaulted[cost_col] >= threshold]

    sample = high_cost if not high_cost.empty else defaulted
    sample = sample.head(15)
    details: list[str] = []
    for _, r in sample.iterrows():
        name = r[name_col] if name_col else "?"
        team = r[team_col] if team_col else "?"
        cost = r[cost_col] if cost_col else "?"
        details.append(f"{name} ({team}, qt_a={cost})")

    log.warning(
        "Role defaulted to 'FWD' for %d player(s) (of which %d high-cost). "
        "Sample: %s. These may be GKs/DEFs mis-classified in formation constraints.",
        int(missing_mask.sum()),
        int(len(high_cost)),
        "; ".join(details) if details else "(no detail columns)",
    )


def _append_foreign_fallback_rows(
    df_player: pd.DataFrame,
    engine: sa.Engine,
    log: logging.Logger,
) -> pd.DataFrame:
    """Append one inference-only row per neo-arrivo with zero Serie A history.

    Prefers ``player_stats_by_prediction_season`` (migration 026) filtered by
    the listino/target season so historical backfill rows are not replaced by
    a newer absolute-latest season. Falls back to
    ``player_latest_stats_any_league`` (migration 018) when the target-aware
    view is unavailable.

    The returned rows are tagged ``is_foreign_fallback=True`` and their
    ``season_start`` is overridden to the domestic pipeline's latest season
    (not the player's real foreign season) so they land in the same
    "latest season" slice used for prediction everywhere downstream. The
    caller (``ml.pipeline.trainer``) is responsible for excluding these rows
    from model training/fitting — this function only adds them.

    Missing view/table (migration not applied) degrades to a no-op with a
    warning, matching the other optional-feature try/except blocks in
    :func:`load_raw_data`.

    Two season numbers matter here and must NOT be conflated:

    - ``listino_season`` — the current Fantacalcio listino season
      (``player_quotations``). This is what ``player_id_map`` /
      ``player_quotations`` are keyed on, and is the season a neo-arrivo
      must be looked up under. Early in a new season the listino for
      season N is imported before Serie A matches for season N have been
      played/scraped, so it can be *ahead* of the domestic data.
    - ``output_season`` — the ``season_start`` written onto the appended
      fallback rows. This MUST equal ``df_player["season_start"].max()``
      (the domestic "latest season" slice), because
      ``ml.pipeline.trainer`` selects the current-squad cohort to project
      forward via ``df[df.season_start == df.season_start.max()]``.
      Tagging fallback rows with the listino season instead would shift
      that max forward and silently drop every normal domestic player
      out of the projection (they only reach last season's data until
      the new season is scraped).

    Using ``listino_season`` for the lookup only, and ``output_season``
    for the label, keeps neo-arrivi discoverable as soon as the listino
    is imported while leaving the domestic prediction cohort untouched.
    """
    if df_player.empty:
        return df_player

    output_season = int(df_player["season_start"].max())
    try:
        listino_season_raw = engine.connect().execute(
            sa.text("SELECT MAX(season_start) FROM player_quotations")
        ).scalar()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not read latest listino season from player_quotations (%s); "
            "falling back to domestic season %d for the foreign fallback lookup.",
            exc, output_season,
        )
        listino_season_raw = None

    listino_season = (
        max(output_season, int(listino_season_raw))
        if listino_season_raw is not None
        else output_season
    )
    if listino_season != output_season:
        log.info(
            "Foreign fallback lookup using listino season %d (domestic Serie A "
            "data only goes up to %d — expected early in a new season before "
            "it has been scraped yet). Appended rows will still be tagged with "
            "season_start=%d to match the current prediction cohort.",
            listino_season, output_season, output_season,
        )

    # Prefer target-aware view (migration 026). If missing, fall back to
    # latest-absolute (migration 018) so environments mid-rollout keep working.
    df_foreign = pd.DataFrame()
    try:
        df_foreign = pd.read_sql(
            sa.text(_FOREIGN_FALLBACK_SQL_TARGET_AWARE),
            engine,
            params={"season_start": listino_season},
        )
        log.debug(
            "Foreign fallback loaded via player_stats_by_prediction_season "
            "(target=%s, rows=%d)",
            listino_season,
            len(df_foreign),
        )
    except Exception as target_exc:  # noqa: BLE001
        log.info(
            "Target-aware foreign view unavailable (%s); "
            "falling back to player_latest_stats_any_league.",
            target_exc,
        )
        try:
            df_foreign = pd.read_sql(
                sa.text(_FOREIGN_FALLBACK_SQL_LATEST),
                engine,
                params={"season_start": listino_season},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not load cross-league neo-arrivo fallback (%s). "
                "Apply migrations 018/026 and run foreign-stats backfill.",
                exc,
            )
            return df_player

    if df_foreign.empty:
        return df_player

    # Exclude only players who already have a DOMESTIC row for the season
    # being predicted (output_season) — NOT "ever had a Serie A row".
    # df_player spans every scraped Serie A season, so a player with an
    # old row (e.g. Serie A 2022-23) but nothing at output_season would
    # otherwise be filtered out here even after the foreign backfill has
    # correctly populated his row (same class of bug as PR8's
    # _candidate_players — "ever played Serie A" vs "has a row now" —
    # just resurfacing in the ML loader instead of the backfill script).
    current_season_ids = set(
        df_player.loc[
            df_player["season_start"] == output_season, "player_fotmob_id"
        ].unique()
    )
    df_foreign = df_foreign[~df_foreign["player_fotmob_id"].isin(current_season_ids)].copy()
    if df_foreign.empty:
        return df_player

    # Back-derive raw counts from the view's pre-computed per-90 rates so the
    # existing add_per90_features() (ml/preprocessing/features.py) reproduces
    # them unchanged — no changes needed there.
    df_foreign["mins_played"] = df_foreign["minutes_avg"]
    denom = (df_foreign["mins_played"] / 90.0).clip(lower=1)
    df_foreign["goals"] = df_foreign["goals_per90"] * denom
    df_foreign["goal_assist"] = df_foreign["assists_per90"] * denom
    df_foreign["saves"] = df_foreign["saves_per90"] * denom
    df_foreign["clean_sheet"] = df_foreign["clean_sheet_per90"] * denom
    df_foreign["appearances"] = (df_foreign["mins_played"] / 90.0).round()
    df_foreign = df_foreign.drop(
        columns=["minutes_avg", "goals_per90", "assists_per90", "saves_per90", "clean_sheet_per90"]
    )

    df_foreign["season_start"] = output_season
    df_foreign["season_label"] = f"{output_season}-foreign-fallback"
    df_foreign["team_fotmob_id"] = pd.NA
    df_foreign["is_foreign_fallback"] = True

    log.info(
        "Appended %d cross-league fallback row(s) for neo-arrivi with zero Serie A history",
        len(df_foreign),
    )
    result = pd.concat([df_player, df_foreign], ignore_index=True, sort=False)
    # team_fotmob_id becomes object dtype once pd.NA is mixed with the Serie A
    # rows' int64 values — pandas then refuses to merge it against the plain
    # int64 key in df_team_strength ("You are trying to merge on object and
    # int64 columns"). Nullable Int64 keeps NA support while staying numeric.
    result["team_fotmob_id"] = result["team_fotmob_id"].astype("Int64")
    return result


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

    # ── Season continuity check ──────────────────────────────────────────────
    # Rolling / delta features (diff(1)) assume one row per season per player.
    # If a season is completely absent (e.g. 2021-22 was not scraped but
    # 2020-21 and 2022-23 are present), the diff spans 2 years instead of 1,
    # creating biased trend features.  We log a warning so the operator knows.
    _seasons_present = sorted(df_player_long["season_start"].unique())
    _season_gaps = []
    for i in range(1, len(_seasons_present)):
        if _seasons_present[i] - _seasons_present[i - 1] > 1:
            _season_gaps.append(
                f"{_seasons_present[i-1]+1}–{_seasons_present[i]-1}"
            )
    if _season_gaps:
        log.warning(
            "SEASON GAP DETECTED: missing season(s) %s between scraped years %s. "
            "Rolling/delta features (diff(1)) will span multi-year gaps instead of 1 — "
            "trend features for affected players will be biased. "
            "Scrape the missing season(s) and re-run to eliminate gaps.",
            ", ".join(_season_gaps),
            _seasons_present,
        )
    else:
        log.info(
            "Season continuity check: %d consecutive season(s) — OK.",
            len(_seasons_present),
        )

    index_cols = [
        "player_fotmob_id", "player_name",
        "team_fotmob_id", "team_name",
        "season_start", "season_label", "league_name",
    ]
    df_player = _pivot_stats(df_player_long, index_cols)
    df_player = canonicalize_columns(df_player)
    df_player = _deduplicate_multi_team_players(df_player)

    # ── Cross-league neo-arrivo fallback (inference-only) ────────────────────
    # Players with zero Serie A history get one extra row from their most
    # recent season in ANY league, so they can still receive a prediction —
    # the trainer must exclude is_foreign_fallback rows from fitting/backtest.
    df_player["is_foreign_fallback"] = False
    if cfg.league_name and cfg.include_foreign_fallback:
        df_player = _append_foreign_fallback_rows(df_player, engine, log)

    # ── Attach player role (season-scoped) ────────────────────────────────────
    # We prefer `player_season_roles` so that each (player, season) row
    # gets the role the player *had* in that season. We fall back to
    # `player_profiles` (current role) if the season-aware table is
    # missing — e.g. before migration 003 has been applied.
    df_player = _attach_role(df_player, engine, log)

    log.info("Loading team_season_stats …")
    df_team_long = pd.read_sql(
        sa.text(_TEAM_STATS_SQL.format(where_clause=where)),
        engine,
    )
    if not df_team_long.empty:
        df_team_strength = _build_team_strength(df_team_long)

        # Log player rows with NULL team_fotmob_id — they won't get team
        # strength features via the left join, so they will be NaN-imputed
        # downstream.  This is expected for players whose team FotMob ID
        # was not scraped (rare edge case).
        _null_team = df_player["team_fotmob_id"].isna().sum()
        if _null_team:
            log.warning(
                "%d player-season rows have NULL team_fotmob_id; "
                "team strength features for them will be NaN (median-imputed).",
                _null_team,
            )

        df_player = df_player.merge(
            df_team_strength,
            on=["team_fotmob_id", "season_start"],
            how="left",
        )
        log.info("  Team strength features merged.")
    else:
        log.warning(
            "No team_season_stats found; team strength features will be NaN. "
            "Run the team-stats scraper step to populate this table."
        )
        # Ensure the expected columns exist with NaN so downstream code
        # (SAP adjuster, imputation) does not crash on missing columns.
        for _col in ("team_strength_score", "is_top_team", "team_rank_norm"):
            df_player[_col] = float("nan")

    # ── Optional: Fantacalcio quotation features ──────────────────────────
    # Best-effort merge against player_quotations / player_id_map. Tables
    # may not exist yet — fall back silently to NaN-only columns.
    try:
        from ..preprocessing.quotation_features import (
            attach_quotation_features as _attach_quot,
        )
        df_player = _attach_quot(df_player, engine=engine)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not attach Fantacalcio quotation features (%s). "
            "Run `python -m ml.data.import_quotations` to populate "
            "player_quotations / player_id_map.",
            exc,
        )
        for col in (
            "qt_a", "qt_i", "qt_a_norm", "qt_i_norm",
            "price_delta_pct", "qt_a_vs_role_median",
            "price_trend_2y", "qt_a_lag1", "match_method",
        ):
            if col not in df_player.columns:
                df_player[col] = pd.NA

    # ── Optional: MANTRA historical features (lagged) ────────────────────
    # Strictly historical: for season N, uses only data from seasons < N.
    try:
        from ..preprocessing.mantra_features import attach_mantra_features
        df_player = attach_mantra_features(df_player, engine=engine)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not attach MANTRA ML features (%s). "
            "Apply migration 016 or ensure player_season_stats is populated.",
            exc,
        )

    log.info("Raw dataset shape: %s", df_player.shape)
    return df_player