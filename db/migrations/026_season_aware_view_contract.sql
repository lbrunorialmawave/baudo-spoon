-- Migration 026: season-aware view / historical target contract (PR5)
--
-- Problem (plan §32):
--   player_latest_stats_any_league returns ONE row per player ordered by
--   absolute season_start DESC. A historical backfill for target=2024 can
--   therefore be shadowed by a newer 2025 row when a consumer reads the
--   latest-only view.
--
-- Contract after this migration:
--   1. player_latest_stats_any_league  — UNCHANGED semantics (latest absolute)
--      for consumers that truly want the single most-recent row.
--   2. player_stats_by_prediction_season — NEW: one row per
--      (fantacalcio_id, prediction_season_start) with explicit source_season_start.
--      Prefer lineage columns from player_season_stats (migration 025) when
--      present; otherwise fall back to seasons.season_start for both.
--
-- Apply:
--   type db/migrations/026_season_aware_view_contract.sql | docker compose exec -T db psql -U fbref -d fbref

-- ── Base aggregates enriched with lineage ───────────────────────────────────
-- Rebuild all-leagues aggregates to expose source/prediction season when set.

CREATE OR REPLACE VIEW player_season_aggregates_all_leagues AS
SELECT
    pss.player_fotmob_id AS fantacalcio_id,
    s.season_start,
    -- Lineage (migration 025). COALESCE keeps BC for pre-lineage rows.
    COALESCE(pss.prediction_season_start, s.season_start) AS prediction_season_start,
    COALESCE(pss.source_season_start, s.season_start)     AS source_season_start,
    MAX(pss.selection_reason) AS selection_reason,
    MAX(pss.fallback_depth)   AS fallback_depth,
    l.name AS league_name,
    AVG(CASE WHEN pss.stat_category = 'mins_played'    THEN pss.value END) AS minutes_avg,
    AVG(CASE WHEN pss.stat_category = 'rating'         THEN pss.value END) AS vote_avg,
    STDDEV(CASE WHEN pss.stat_category = 'rating' THEN pss.value END)      AS vote_std,
    LEAST(AVG(CASE WHEN pss.stat_category = 'mins_played' THEN pss.value END) / 3420.0, 1.0) AS presence_rate,
    AVG(CASE WHEN pss.stat_category = 'expected_goals_per_90'   THEN pss.value END) AS xg_per90,
    AVG(CASE WHEN pss.stat_category = 'expected_assists_per_90' THEN pss.value END) AS xa_per90,
    AVG(CASE WHEN pss.stat_category = 'goals_per_90'            THEN pss.value END) AS goals_per90,
    AVG(CASE WHEN pss.stat_category = 'goal_assist'             THEN pss.value END) AS assists_per90,
    AVG(CASE WHEN pss.stat_category = 'saves'                   THEN pss.value END) AS saves_per90,
    AVG(CASE WHEN pss.stat_category = 'clean_sheet'             THEN pss.value END) AS clean_sheet_per90
FROM player_season_stats pss
JOIN seasons s ON s.id = pss.season_id
JOIN leagues l ON l.id = s.league_id
GROUP BY
    pss.player_fotmob_id,
    s.season_start,
    COALESCE(pss.prediction_season_start, s.season_start),
    COALESCE(pss.source_season_start, s.season_start),
    l.name;

-- ── Latest-absolute view (BC — one row per player) ──────────────────────────
CREATE OR REPLACE VIEW player_latest_stats_any_league AS
SELECT
    fantacalcio_id,
    season_start,
    prediction_season_start,
    source_season_start,
    selection_reason,
    fallback_depth,
    league_name,
    minutes_avg,
    vote_avg,
    vote_std,
    presence_rate,
    xg_per90,
    xa_per90,
    goals_per90,
    assists_per90,
    saves_per90,
    clean_sheet_per90
FROM (
    SELECT a.*,
           ROW_NUMBER() OVER (
               PARTITION BY a.fantacalcio_id
               ORDER BY a.season_start DESC
           ) AS rn
    FROM player_season_aggregates_all_leagues a
) ranked
WHERE rn = 1;

-- ── Target-aware view: one row per (player, prediction_season) ──────────────
-- Prefer the row whose source is closest to the prediction season when
-- multiple source rows share the same prediction_season_start (should be rare).
CREATE OR REPLACE VIEW player_stats_by_prediction_season AS
SELECT
    fantacalcio_id,
    prediction_season_start,
    source_season_start,
    season_start,
    selection_reason,
    fallback_depth,
    league_name,
    minutes_avg,
    vote_avg,
    vote_std,
    presence_rate,
    xg_per90,
    xa_per90,
    goals_per90,
    assists_per90,
    saves_per90,
    clean_sheet_per90
FROM (
    SELECT a.*,
           ROW_NUMBER() OVER (
               PARTITION BY a.fantacalcio_id, a.prediction_season_start
               ORDER BY a.source_season_start DESC
           ) AS rn
    FROM player_season_aggregates_all_leagues a
) ranked
WHERE rn = 1;

COMMENT ON VIEW player_latest_stats_any_league IS
    'Latest-absolute stats per player (any league). Not suitable alone for historical target lookups.';

COMMENT ON VIEW player_stats_by_prediction_season IS
    'Season-aware stats: one row per (player, prediction_season_start). '
    'Use WHERE prediction_season_start = :target for historical/backfill consumers.';
