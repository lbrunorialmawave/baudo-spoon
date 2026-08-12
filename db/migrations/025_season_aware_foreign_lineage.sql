-- Migration 025: season-aware foreign lineage
--
-- Adds explicit source / prediction season columns on player_season_stats so
-- foreign-fallback rows can answer:
--   "which season did the stats come from?"
--   "which prediction/target season are they being used for?"
--
-- source_season_start is largely derivable via season_id → seasons.season_start
-- but is denormalised for query/audit convenience and for rows where the
-- logical source differs from the Season row used for persistence.
--
-- prediction_season_start is genuinely new lineage: the season the row is
-- intended to serve (may differ from source when previous-season fallback is
-- used during a current-season refresh).
--
-- fotmob_season_id = -1 remains the documented sentinel for foreign career
-- snapshots that do not come from a bulk league scrape (see
-- scraper/src/player_career_scraper._persist_one_snapshot).
--
-- Backward compatible: both new columns are nullable; existing rows keep NULL
-- until a future backfill or re-ingest decides otherwise.
--
-- Apply:
--   type db/migrations/025_season_aware_foreign_lineage.sql | docker compose exec -T db psql -U fbref -d fbref

ALTER TABLE player_season_stats
    ADD COLUMN IF NOT EXISTS source_season_start INTEGER NULL;

ALTER TABLE player_season_stats
    ADD COLUMN IF NOT EXISTS prediction_season_start INTEGER NULL;

ALTER TABLE player_season_stats
    ADD COLUMN IF NOT EXISTS selection_reason VARCHAR(64) NULL;

ALTER TABLE player_season_stats
    ADD COLUMN IF NOT EXISTS fallback_depth SMALLINT NULL;

COMMENT ON COLUMN player_season_stats.source_season_start IS
    'Season year the stats were taken from (FotMob careerHistory entry). '
    'May equal seasons.season_start; denormalised for audit and target-aware queries.';

COMMENT ON COLUMN player_season_stats.prediction_season_start IS
    'Season year this row is intended to serve for prediction/backfill. '
    'Differs from source_season_start when previous-season fallback is used.';

COMMENT ON COLUMN player_season_stats.selection_reason IS
    'SeasonResolutionResult.reason code (e.g. TARGET_SEASON_SELECTED, PREVIOUS_SEASON_SELECTED).';

COMMENT ON COLUMN player_season_stats.fallback_depth IS
    'How many seasons were walked back from the target (0 = exact target).';

COMMENT ON COLUMN player_season_stats.fotmob_season_id IS
    'FotMob season id from bulk scrape; sentinel -1 for foreign careerHistory snapshots.';

-- Helpful for target-aware lookups (PR5 will lean on this).
CREATE INDEX IF NOT EXISTS idx_pss_prediction_season
    ON player_season_stats (prediction_season_start)
    WHERE prediction_season_start IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pss_source_season
    ON player_season_stats (source_season_start)
    WHERE source_season_start IS NOT NULL;
