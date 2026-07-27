-- Migration: add manual_resolutions table
--
-- Motivation: every time a new scrape or import_quotations re-runs,
-- manually resolved Fantacalcio ↔ FotMob associations can be lost or
-- overwritten.  This table records each manual resolution permanently,
-- and the matching pipeline consults it as "Pass 0" — before any
-- automatic matching — so that once the operator resolves a player,
-- the mapping is never forgotten.
--
-- Key design decisions:
--   * Standalone table (not a column on player_id_map) because the
--     history is immutable and cross-season; player_id_map is per-season
--     and overwritable.
--   * UNIQUE (fantacalcio_id, player_fotmob_id) prevents duplicate
--     entries for the same association.
--   * No FK to player_id_map — deliberately independent so that
--     resolutions survive even if player_id_map is dropped/rebuilt.
--   * ``resolved_by`` is nullable for now (auth not fully integrated
--     in the mapping pipeline yet).
--
-- Apply:
--   docker compose exec -T db psql -U fbref -d fbref \
--       -f /docker-entrypoint-initdb.d/013_add_manual_resolutions.sql

CREATE TABLE IF NOT EXISTS manual_resolutions (
    id               BIGSERIAL PRIMARY KEY,

    -- Fantacalcio ID that was resolved (stable cross-season key).
    fantacalcio_id   INTEGER NOT NULL,

    -- FotMob ID assigned by the operator.
    player_fotmob_id BIGINT  NOT NULL,

    -- Season in which this resolution was made.
    season_start     INTEGER NOT NULL,

    -- Snapshot of the data at resolution time (informational / audit).
    name_fantacalcio VARCHAR(200) NOT NULL,
    team_fantacalcio VARCHAR(100),
    canonical_role   VARCHAR(5),
    name_fotmob      VARCHAR(200),
    team_fotmob      VARCHAR(100),

    -- Who resolved it and why (optional).
    resolved_by      VARCHAR(100),
    note             TEXT,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Same association should not be recorded twice.
    UNIQUE (fantacalcio_id, player_fotmob_id)
);

-- Indexes for fast lookups and stats.
CREATE INDEX IF NOT EXISTS idx_mr_fantacalcio
    ON manual_resolutions (fantacalcio_id);

CREATE INDEX IF NOT EXISTS idx_mr_fotmob
    ON manual_resolutions (player_fotmob_id);

CREATE INDEX IF NOT EXISTS idx_mr_season
    ON manual_resolutions (season_start);

CREATE INDEX IF NOT EXISTS idx_mr_created
    ON manual_resolutions (created_at DESC);

COMMENT ON TABLE manual_resolutions IS
    'Permanent record of manually resolved Fantacalcio ↔ FotMob ID associations.';

COMMENT ON COLUMN manual_resolutions.fantacalcio_id IS
    'Fantacalcio ID from the listone XLSX (stable across seasons).';

COMMENT ON COLUMN manual_resolutions.player_fotmob_id IS
    'FotMob player ID assigned by the operator.';

COMMENT ON COLUMN manual_resolutions.season_start IS
    'Season in which this resolution was recorded.';

COMMENT ON COLUMN manual_resolutions.name_fantacalcio IS
    'Player name as it appeared in the Fantacalcio listone at resolution time.';

COMMENT ON COLUMN manual_resolutions.team_fantacalcio IS
    'Team as it appeared in the Fantacalcio listone at resolution time.';

COMMENT ON COLUMN manual_resolutions.canonical_role IS
    'Canonical role (GK/DEF/MID/FWD) at resolution time.';

COMMENT ON COLUMN manual_resolutions.name_fotmob IS
    'FotMob player name at resolution time.';

COMMENT ON COLUMN manual_resolutions.team_fotmob IS
    'FotMob team name at resolution time.';

COMMENT ON COLUMN manual_resolutions.resolved_by IS
    'Who performed the resolution (user email / ID). Nullable until auth is integrated.';

COMMENT ON COLUMN manual_resolutions.note IS
    'Free-text note explaining why this resolution was made.';
