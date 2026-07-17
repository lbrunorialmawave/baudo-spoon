-- Migration: create aggregate views required by the MANTRA scoring runner
--
-- The MANTRA runner queries player_season_aggregates and
-- team_strength_aggregates, which pivot the row-per-stat rows
-- from player_season_stats / team_season_stats into wide columns.
--
-- Apply:
--   type db\migrations\006_create_mantra_views.sql | docker compose exec -T db psql -U fbref -d fbref

CREATE OR REPLACE VIEW player_season_aggregates AS
SELECT
    pss.player_fotmob_id AS fantacalcio_id,
    s.season_start,
    AVG(CASE WHEN pss.stat_category = 'mins_played'    THEN pss.value END) AS minutes_avg,
    AVG(CASE WHEN pss.stat_category = 'rating'         THEN pss.value END) AS vote_avg,
    STDDEV(CASE WHEN pss.stat_category = 'rating' THEN pss.value END)      AS vote_std,
    COUNT(CASE WHEN pss.stat_category = 'mins_played' AND pss.value > 0 THEN 1 END)
        / NULLIF(COUNT(DISTINCT pss.stat_category), 0)                     AS presence_rate,
    AVG(CASE WHEN pss.stat_category = 'expected_goals_per_90'       THEN pss.value END) AS xg_per90,
    AVG(CASE WHEN pss.stat_category = 'expected_assists_per_90'     THEN pss.value END) AS xa_per90,
    AVG(CASE WHEN pss.stat_category = 'goals_per_90'               THEN pss.value END) AS goals_per90,
    AVG(CASE WHEN pss.stat_category = 'goal_assist'                THEN pss.value END) AS assists_per90,
    AVG(CASE WHEN pss.stat_category = 'saves'                      THEN pss.value END) AS saves_per90,
    AVG(CASE WHEN pss.stat_category = 'clean_sheet'                THEN pss.value END) AS clean_sheet_per90,
    COUNT(DISTINCT s.season_start)                                                     AS seasons_in_italy
FROM player_season_stats pss
JOIN seasons s ON s.id = pss.season_id
GROUP BY pss.player_fotmob_id, s.season_start;

CREATE OR REPLACE VIEW team_strength_aggregates AS
SELECT
    tss.team_name,
    s.season_start,
    AVG(CASE WHEN tss.stat_category = 'rating'         THEN tss.value END) AS team_rank_norm,
    AVG(CASE WHEN tss.stat_category = 'goals'          THEN tss.value END) AS prev_season_points,
    AVG(CASE WHEN tss.stat_category = 'goals_conceded'  THEN tss.value END) AS goal_difference,
    AVG(CASE WHEN tss.stat_category = 'rating'         THEN tss.value END) AS avg_team_rating
FROM team_season_stats tss
JOIN seasons s ON s.id = tss.season_id
GROUP BY tss.team_name, s.season_start;
