-- Migration: add player_matchday_status table for matchday-specific data.
--
-- Stores per-player status for each matchday: probable starter probability,
-- injury/suspension info, and tactical ballots. Populated by the
-- probabili_formazioni scraper.

CREATE TABLE IF NOT EXISTS player_matchday_status (
    id               BIGSERIAL    PRIMARY KEY,
    fantacalcio_id   INT          NOT NULL,
    season_start     INT          NOT NULL,
    matchday         INT          NOT NULL,
    team             VARCHAR(100) NOT NULL,
    probability      SMALLINT     NOT NULL DEFAULT 0,       -- 0-100 %
    status           VARCHAR(20)  NOT NULL DEFAULT 'unknown',

    injury_note      TEXT,                                   -- injury description
    scraped_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_player_matchday UNIQUE (fantacalcio_id, season_start, matchday)
);

COMMENT ON TABLE  player_matchday_status IS
    'Matchday-specific player status: probable starter probability, injuries, suspensions.';
COMMENT ON COLUMN player_matchday_status.probability IS
    'Estimated probability (0-100) of being in the starting XI.';
COMMENT ON COLUMN player_matchday_status.status IS
    'Current status: starter, bench, injured, suspended, doubtful, unknown.';

CREATE INDEX IF NOT EXISTS idx_pms_matchday  ON player_matchday_status (matchday);
CREATE INDEX IF NOT EXISTS idx_pms_season    ON player_matchday_status (season_start);
CREATE INDEX IF NOT EXISTS idx_pms_team      ON player_matchday_status (team);
CREATE INDEX IF NOT EXISTS idx_pms_status    ON player_matchday_status (status);
