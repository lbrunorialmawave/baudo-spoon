-- Migration: scope MANTRA views to Serie A + add cross-league fallback views
--
-- Preparatory work for ingesting foreign leagues (Premier League, La Liga,
-- Bundesliga, Ligue 1 — already catalogued in scraper/src/models.py) so a
-- player brand-new to Serie A can use his last real season's stats (even if
-- abroad) instead of a role-median guess, without corrupting existing
-- Serie-A-only semantics:
--
-- 1. player_season_aggregates gets a Serie A filter. Today it has none —
--    harmless while only Serie A is scraped, but the moment a second league
--    shares a season_start with Serie A, seasons_in_italy (COUNT of prior
--    seasons) and every stat column would silently blend both leagues.
--    seasons_in_italy specifically must stay Serie A-only: it feeds
--    is_neo_arrivo in ml/mantra/pilastro1.py, which encodes "adaptation
--    risk to Italian football" — a real, separate fact from "has stats
--    somewhere" that must not be erased by adding foreign data.
--
-- 2. team_strength_aggregates gains a league_name column. It's joined in
--    ml/mantra/runner.py by team_name string (not team_fotmob_id) — no
--    collision exists today between Serie A and big-5 club names, but
--    adding 4x more clubs into the same table without a league filter is
--    fragile. ml/mantra/runner.py is updated in the same change to filter
--    both joins to Serie A.
--
-- 3. Two new views support the actual fallback: player_season_aggregates_all_leagues
--    (per player/season/league, unfiltered) and player_latest_stats_any_league
--    (one row per player: his single most recent season, any league). MANTRA's
--    runner adds this as a THIRD COALESCE tier — used only when the player has
--    no Serie A row for the target season or the one before it.
--
-- Apply:
--   type db\migrations\018_scope_mantra_views_and_add_foreign_fallback.sql | docker compose exec -T db psql -U fbref -d fbref

-- ── 1. Serie-A-scope player_season_aggregates (structure matches 017's fix) ──

DROP VIEW IF EXISTS player_season_aggregates;

CREATE VIEW player_season_aggregates AS
WITH per_season AS (
    SELECT
        pss.player_fotmob_id AS fantacalcio_id,
        s.season_start,
        AVG(CASE WHEN pss.stat_category = 'mins_played'    THEN pss.value END) AS minutes_avg,
        AVG(CASE WHEN pss.stat_category = 'rating'         THEN pss.value END) AS vote_avg,
        STDDEV(CASE WHEN pss.stat_category = 'rating' THEN pss.value END)      AS vote_std,
        AVG(CASE WHEN pss.stat_category = 'expected_goals_per_90'       THEN pss.value END) AS xg_per90,
        AVG(CASE WHEN pss.stat_category = 'expected_assists_per_90'     THEN pss.value END) AS xa_per90,
        AVG(CASE WHEN pss.stat_category = 'goals_per_90'               THEN pss.value END) AS goals_per90,
        AVG(CASE WHEN pss.stat_category = 'goal_assist'                THEN pss.value END) AS assists_per90,
        AVG(CASE WHEN pss.stat_category = 'saves'                      THEN pss.value END) AS saves_per90,
        AVG(CASE WHEN pss.stat_category = 'clean_sheet'                THEN pss.value END) AS clean_sheet_per90
    FROM player_season_stats pss
    JOIN seasons s ON s.id = pss.season_id
    JOIN leagues l ON l.id = s.league_id
    WHERE l.name = 'Serie A'
    GROUP BY pss.player_fotmob_id, s.season_start
)
SELECT
    cur.fantacalcio_id,
    cur.season_start,
    cur.minutes_avg,
    cur.vote_avg,
    cur.vote_std,
    LEAST(cur.minutes_avg / 3420.0, 1.0)                                       AS presence_rate,
    cur.xg_per90,
    cur.xa_per90,
    cur.goals_per90,
    cur.assists_per90,
    cur.saves_per90,
    cur.clean_sheet_per90,
    COUNT(prior.season_start)                                                  AS seasons_in_italy
FROM per_season cur
LEFT JOIN per_season prior
    ON prior.fantacalcio_id = cur.fantacalcio_id
    AND prior.season_start < cur.season_start
GROUP BY
    cur.fantacalcio_id, cur.season_start, cur.minutes_avg, cur.vote_avg, cur.vote_std,
    cur.xg_per90, cur.xa_per90, cur.goals_per90, cur.assists_per90,
    cur.saves_per90, cur.clean_sheet_per90;

-- ── 2. Add league_name to team_strength_aggregates ──────────────────────────
-- DROP+CREATE rather than CREATE OR REPLACE: Postgres only allows REPLACE to
-- append trailing columns, not insert one in the middle of the column list.

DROP VIEW IF EXISTS team_strength_aggregates;

CREATE VIEW team_strength_aggregates AS
SELECT
    tss.team_name,
    s.season_start,
    l.name AS league_name,
    AVG(CASE WHEN tss.stat_category = 'rating'         THEN tss.value END) AS team_rank_norm,
    AVG(CASE WHEN tss.stat_category = 'goals'          THEN tss.value END) AS prev_season_points,
    AVG(CASE WHEN tss.stat_category = 'goals_conceded'  THEN tss.value END) AS goal_difference,
    AVG(CASE WHEN tss.stat_category = 'rating'         THEN tss.value END) AS avg_team_rating
FROM team_season_stats tss
JOIN seasons s ON s.id = tss.season_id
JOIN leagues l ON l.id = s.league_id
GROUP BY tss.team_name, s.season_start, l.name;

-- ── 3. Cross-league fallback views (no Serie A filter — this is the point) ──

CREATE OR REPLACE VIEW player_season_aggregates_all_leagues AS
SELECT
    pss.player_fotmob_id AS fantacalcio_id,
    s.season_start,
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
GROUP BY pss.player_fotmob_id, s.season_start, l.name;

CREATE OR REPLACE VIEW player_latest_stats_any_league AS
SELECT fantacalcio_id, season_start, league_name, minutes_avg, vote_avg, vote_std,
       presence_rate, xg_per90, xa_per90, goals_per90, assists_per90,
       saves_per90, clean_sheet_per90
FROM (
    SELECT a.*,
           ROW_NUMBER() OVER (PARTITION BY a.fantacalcio_id ORDER BY a.season_start DESC) AS rn
    FROM player_season_aggregates_all_leagues a
) ranked
WHERE rn = 1;
