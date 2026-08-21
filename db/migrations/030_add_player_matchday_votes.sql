-- Migration: add player_matchday_votes for per-matchday Fantacalcio grades.
--
-- Populated by the Node scraper (voti/scraper.js) + Python loader
-- (ml/data/voti_matchday_loader.py). Used by the Trade Fairness Engine
-- to compute Forma_Recente (EWMA of recent matchday scores).
--
-- Design notes:
-- * One row per (player, season, matchday, fonte). The three sources
--   (fantacalcio / statistico / italia) are stored side-by-side so the
--   fairness engine can pick the preferred metric without re-scraping.
-- * fantacalcio_id is the stable Fantacalcio catalog id (same key used
--   by player_matchday_status and expert_ratings).
-- * Bonus/malus counts are denormalised for convenience; the raw JSON
--   from the scraper remains the source of truth if finer detail is needed.

CREATE TABLE IF NOT EXISTS player_matchday_votes (
    id                  BIGSERIAL    PRIMARY KEY,
    fantacalcio_id      INT          NOT NULL,
    season_start        INT          NOT NULL,
    giornata            INT          NOT NULL,
    team                VARCHAR(100),
    ruolo               VARCHAR(20),                 -- Portiere / Difensore / ...

    -- Preferred metric (fantacalcio source) kept at top level for fast reads
    voto_fantacalcio    NUMERIC(4,2),                -- 0–10 or NULL (s.v.)
    fantavoto           NUMERIC(5,2),                -- includes bonus/malus

    -- Parallel sources (nullable when the site only published one grade)
    voto_statistico     NUMERIC(4,2),
    fantavoto_statistico NUMERIC(5,2),
    voto_italia         NUMERIC(4,2),
    fantavoto_italia    NUMERIC(5,2),

    -- Aggregated event counts (from bonus/malus icons)
    gol                 SMALLINT     NOT NULL DEFAULT 0,
    assist              SMALLINT     NOT NULL DEFAULT 0,
    ammonizioni         SMALLINT     NOT NULL DEFAULT 0,
    espulsioni          SMALLINT     NOT NULL DEFAULT 0,
    gol_subiti          SMALLINT     NOT NULL DEFAULT 0,
    rigori_parati       SMALLINT     NOT NULL DEFAULT 0,
    rigori_sbagliati    SMALLINT     NOT NULL DEFAULT 0,

    -- Provenance
    fonte               VARCHAR(20)  NOT NULL DEFAULT 'fantacalcio',
    scraped_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_player_matchday_votes
        UNIQUE (fantacalcio_id, season_start, giornata, fonte)
);

COMMENT ON TABLE  player_matchday_votes IS
    'Per-matchday Fantacalcio grades (voto + fantavoto) for Forma_Recente EWMA.';
COMMENT ON COLUMN player_matchday_votes.voto_fantacalcio IS
    'Raw vote from fantacalcio.it (NULL when s.v. / did not play).';
COMMENT ON COLUMN player_matchday_votes.fantavoto IS
    'Fantavoto including bonus/malus from the fantacalcio source.';
COMMENT ON COLUMN player_matchday_votes.fonte IS
    'Source label: fantacalcio | statistico | italia (row-level when stored separately).';

CREATE INDEX IF NOT EXISTS idx_pmv_season_giornata
    ON player_matchday_votes (season_start, giornata);
CREATE INDEX IF NOT EXISTS idx_pmv_player_season
    ON player_matchday_votes (fantacalcio_id, season_start);
CREATE INDEX IF NOT EXISTS idx_pmv_team
    ON player_matchday_votes (team);
