-- Migration: cumulative MANTRA feature view for ML training
--
-- This view computes per-player CUMULATIVE stats up to (but excluding)
-- each season, enabling lag-safe ML features: for a row with season_start=N,
-- all stats are computed from seasons < N only → zero leakage.
--
-- Apply:
--   type db\migrations\016_add_mantra_ml_features_view.sql | docker compose exec -T db psql -U fbref -d fbref

CREATE OR REPLACE VIEW player_mantra_ml_features AS
WITH per_season AS (
    SELECT
        pss.player_fotmob_id,
        s.season_start,
        AVG(CASE WHEN pss.stat_category = 'rating' THEN pss.value END)                 AS vote_avg,
        STDDEV(CASE WHEN pss.stat_category = 'rating' THEN pss.value END)              AS vote_std,
        AVG(CASE WHEN pss.stat_category = 'mins_played' THEN pss.value END)            AS minutes_avg,
        AVG(CASE WHEN pss.stat_category = 'expected_goals_per_90' THEN pss.value END)  AS xg_per90,
        AVG(CASE WHEN pss.stat_category = 'expected_assists_per_90' THEN pss.value END) AS xa_per90,
        COUNT(CASE WHEN pss.stat_category = 'mins_played' AND pss.value > 0 THEN 1 END)::float
            / NULLIF(COUNT(DISTINCT pss.stat_category), 0)                              AS presence_rate
    FROM player_season_stats pss
    JOIN seasons s ON s.id = pss.season_id
    GROUP BY pss.player_fotmob_id, s.season_start
),
cumulative AS (
    SELECT
        ps.player_fotmob_id,
        ps.season_start,
        -- Cumulative averages from ALL prior seasons (< current)
        AVG(prior.vote_avg)       OVER w AS mantra_vote_avg,
        AVG(prior.vote_std)       OVER w AS mantra_vote_std,
        AVG(prior.minutes_avg)    OVER w AS mantra_minutes_avg,
        AVG(prior.xg_per90)       OVER w AS mantra_xg_per90,
        AVG(prior.xa_per90)       OVER w AS mantra_xa_per90,
        AVG(prior.presence_rate)  OVER w AS mantra_presence_rate,
        COUNT(prior.season_start) OVER w AS mantra_seasons_it
    FROM per_season ps
    LEFT JOIN per_season prior
        ON  prior.player_fotmob_id = ps.player_fotmob_id
        AND prior.season_start < ps.season_start
    WINDOW w AS (PARTITION BY ps.player_fotmob_id, ps.season_start)
)
SELECT DISTINCT
    player_fotmob_id,
    season_start,
    mantra_vote_avg,
    mantra_vote_std,
    mantra_minutes_avg,
    mantra_xg_per90,
    mantra_xa_per90,
    mantra_presence_rate,
    mantra_seasons_it
FROM cumulative;
