-- Migration 029: add league_comp_id and league_country_code to team_strength_aggregates
--
-- runner.py already joins on ts.league_comp_id = '55' (Phase 5 identity fix)
-- but the view only exposed league_name. Adding the two identity columns at the
-- end satisfies CREATE OR REPLACE requirements (no existing column is moved).
--
-- Apply:
--   psql -d defaultdb -f db/migrations/029_team_strength_aggregates_comp_id.sql

CREATE OR REPLACE VIEW team_strength_aggregates AS
SELECT
    tss.team_name,
    s.season_start,
    l.name AS league_name,
    AVG(CASE WHEN tss.stat_category = 'rating'          THEN tss.value END) AS team_rank_norm,
    AVG(CASE WHEN tss.stat_category = 'goals'           THEN tss.value END) AS prev_season_points,
    AVG(CASE WHEN tss.stat_category = 'goals_conceded'  THEN tss.value END) AS goal_difference,
    AVG(CASE WHEN tss.stat_category = 'rating'          THEN tss.value END) AS avg_team_rating,
    l.comp_id       AS league_comp_id,
    l.country_code  AS league_country_code
FROM team_season_stats tss
JOIN seasons s ON s.id = tss.season_id
JOIN leagues l ON l.id = s.league_id
GROUP BY tss.team_name, s.season_start, l.name, l.comp_id, l.country_code;
