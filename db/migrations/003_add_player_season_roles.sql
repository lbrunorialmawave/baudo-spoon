-- Migration: add season-aware player role lookup.
--
-- Rationale: `player_profiles` is keyed solely on `player_fotmob_id`, so a
-- re-scrape (or a FotMob role reclassification) overwrites the historical
-- role of every player. The ML pipeline joins on `player_fotmob_id` only,
-- which means feature rows from past seasons can be silently relabelled
-- to the *current* role, breaking temporal consistency.
--
-- This table preserves the role per (player, season) and becomes the
-- source of truth for the ML loader. `player_profiles` is kept as the
-- "current/canonical" state for fast lookups (e.g. UI).
--
-- Backward-compatible: does not alter any existing table. Safe to apply
-- while the scraper is running — it is `CREATE TABLE IF NOT EXISTS`.
--
-- Apply once:
--   psql -U <user> -d <db> -f 003_add_player_season_roles.sql

CREATE TABLE IF NOT EXISTS player_season_roles (
    player_fotmob_id  BIGINT       NOT NULL,
    season_start      INT          NOT NULL,
    role_key          VARCHAR(50),
    canonical_role    VARCHAR(5)
        CHECK (canonical_role IN ('GK', 'DEF', 'MID', 'FWD')),
    source            VARCHAR(20)  NOT NULL DEFAULT 'fotmob',
    scraped_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_fotmob_id, season_start),
    CONSTRAINT fk_psr_player
        FOREIGN KEY (player_fotmob_id)
        REFERENCES player_profiles (player_fotmob_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psr_season
    ON player_season_roles (season_start);

CREATE INDEX IF NOT EXISTS idx_psr_canonical_role
    ON player_season_roles (canonical_role);
