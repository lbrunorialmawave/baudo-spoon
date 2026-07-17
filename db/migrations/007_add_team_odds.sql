-- Migration: add team_season_odds table for pre-season betting odds.
--
-- Stores win probability for each team per season from various bookmakers
-- (Snai, bet365, etc.). Used as a weight in the PS_corretto computation
-- (Pilastro 3 — Peso Squadra).

CREATE TABLE IF NOT EXISTS team_season_odds (
    id                   BIGSERIAL    PRIMARY KEY,
    team                 VARCHAR(100) NOT NULL,
    season_start         INT          NOT NULL,
    odds                 NUMERIC(8,2) NOT NULL,
    implied_probability  NUMERIC(5,2) NOT NULL,  -- 0-100, already normalised
    source               VARCHAR(50)  NOT NULL DEFAULT 'snai',
    scraped_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_team_season_odds UNIQUE (team, season_start, source)
);

COMMENT ON TABLE  team_season_odds IS
    'Pre-season betting odds converted to implied probabilities per team-season.';
COMMENT ON COLUMN team_season_odds.implied_probability IS
    'Implied win probability (0-100) normalised to remove bookmaker overround.';

CREATE INDEX IF NOT EXISTS idx_tso_season ON team_season_odds (season_start);
CREATE INDEX IF NOT EXISTS idx_tso_team   ON team_season_odds (team);
