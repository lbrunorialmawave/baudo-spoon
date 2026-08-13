-- Migration 028: retire name-as-identity for leagues; extend views with league identity cols
--
-- Phase 4: swap the unique constraint that governs league identity.
--   Before: UNIQUE(name) — fails when two different leagues share a display name
--           (e.g. Serie A ITA comp_id=55, Serie A BRA comp_id=268).
--   After:  UNIQUE(comp_id) WHERE comp_id IS NOT NULL  — already in 027
--           UNIQUE(name)    WHERE comp_id IS NULL       — uncatalogued rows stay name-unique
--
-- Phase 5 (view): add league_id, league_comp_id, league_country_code to the
--   three views so ML consumers can distinguish competitions without string parsing.
--
-- Apply:
--   psql -d defaultdb -f db/migrations/028_league_comp_id_identity_and_view_extension.sql

-- ── Phase 4: swap unique constraint ─────────────────────────────────────────

-- 1. Partial unique for uncatalogued leagues: name must be unique within the
--    set of rows that have no comp_id yet.
CREATE UNIQUE INDEX IF NOT EXISTS uq_leagues_name_uncatalogued
    ON leagues (name)
    WHERE comp_id IS NULL;

-- 2. Drop the old full unique constraint — replaced by the two partial indexes.
ALTER TABLE leagues DROP CONSTRAINT IF EXISTS leagues_name_key;

-- ── Phase 5: extend views with league identity columns ───────────────────────

CREATE OR REPLACE VIEW player_season_aggregates_all_leagues AS
SELECT
    pss.player_fotmob_id AS fantacalcio_id,
    s.season_start,
    COALESCE(pss.prediction_season_start, s.season_start) AS prediction_season_start,
    COALESCE(pss.source_season_start, s.season_start)     AS source_season_start,
    MAX(pss.selection_reason) AS selection_reason,
    MAX(pss.fallback_depth)   AS fallback_depth,
    l.name          AS league_name,
    AVG(CASE WHEN pss.stat_category = 'mins_played'    THEN pss.value END) AS minutes_avg,
    AVG(CASE WHEN pss.stat_category = 'rating'         THEN pss.value END) AS vote_avg,
    STDDEV(CASE WHEN pss.stat_category = 'rating' THEN pss.value END)      AS vote_std,
    LEAST(AVG(CASE WHEN pss.stat_category = 'mins_played' THEN pss.value END) / 3420.0, 1.0) AS presence_rate,
    AVG(CASE WHEN pss.stat_category = 'expected_goals_per_90'   THEN pss.value END) AS xg_per90,
    AVG(CASE WHEN pss.stat_category = 'expected_assists_per_90' THEN pss.value END) AS xa_per90,
    AVG(CASE WHEN pss.stat_category = 'goals_per_90'            THEN pss.value END) AS goals_per90,
    AVG(CASE WHEN pss.stat_category = 'goal_assist'             THEN pss.value END) AS assists_per90,
    AVG(CASE WHEN pss.stat_category = 'saves'                   THEN pss.value END) AS saves_per90,
    AVG(CASE WHEN pss.stat_category = 'clean_sheet'             THEN pss.value END) AS clean_sheet_per90,
    -- League identity (migration 028): new columns at end for CREATE OR REPLACE compatibility.
    l.id            AS league_id,
    l.comp_id       AS league_comp_id,
    l.country_code  AS league_country_code
FROM player_season_stats pss
JOIN seasons s ON s.id = pss.season_id
JOIN leagues l ON l.id = s.league_id
GROUP BY
    pss.player_fotmob_id,
    s.season_start,
    COALESCE(pss.prediction_season_start, s.season_start),
    COALESCE(pss.source_season_start, s.season_start),
    l.id,
    l.name,
    l.comp_id,
    l.country_code;

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
    clean_sheet_per90,
    league_id,
    league_comp_id,
    league_country_code
FROM (
    SELECT a.*,
           ROW_NUMBER() OVER (
               PARTITION BY a.fantacalcio_id
               ORDER BY a.season_start DESC
           ) AS rn
    FROM player_season_aggregates_all_leagues a
) ranked
WHERE rn = 1;

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
    clean_sheet_per90,
    league_id,
    league_comp_id,
    league_country_code
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
    'Filter by prediction_season_start for historical/backfill consumers. '
    'league_comp_id and league_country_code distinguish homonymous competitions.';
