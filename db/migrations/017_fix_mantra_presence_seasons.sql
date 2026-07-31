-- Migration: fix player_season_aggregates (seasons_in_italy always 1, broken presence_rate)
--
-- Two bugs in the view created by 010_create_mantra_views.sql:
--
-- 1. seasons_in_italy = COUNT(DISTINCT s.season_start) inside a
--    GROUP BY ... s.season_start always evaluates to 1 (the group only ever
--    contains one season_start value). Fase 7's CERTEZZA rule requires
--    Stagioni_IT >= 2, so it could never fire. Fixed by counting, per
--    player/season, how many STRICTLY EARLIER seasons that player has data
--    for (self-join), matching the cumulative pattern already used in
--    016_add_mantra_ml_features_view.sql for mantra_seasons_it.
--
-- 2. presence_rate counted "how many distinct stat categories are tracked"
--    instead of "fraction of the season actually played". Fixed as
--    minutes_avg / 3420 (38 matchdays x 90'), the same maximum used
--    elsewhere in the codebase (e.g. ml/mantra/config.py SOGLIA_MINUTI_MAX
--    context, ml/targets/builder.py's mins_played/(appearances*90)).
--    3420 is also the observed MAX(mins_played) in player_season_stats.
--
-- presence_rate changes type from bigint (old integer-division bug: values
-- were only ever exactly 0 or 1) to numeric (a real 0-1 fraction), so the
-- view must be dropped and recreated rather than CREATE OR REPLACE'd.
--
-- Apply:
--   type db\migrations\017_fix_mantra_presence_seasons.sql | docker compose exec -T db psql -U fbref -d fbref

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
