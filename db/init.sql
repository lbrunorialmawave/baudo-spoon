-- FotMob Data Platform — PostgreSQL schema
-- Executed automatically on first container start.

CREATE TABLE IF NOT EXISTS leagues (
    id          SERIAL       PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    comp_id     VARCHAR(10)  NOT NULL,
    slug        VARCHAR(200) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    id           SERIAL      PRIMARY KEY,
    league_id    INT         NOT NULL REFERENCES leagues(id) ON DELETE CASCADE,
    season_start INT         NOT NULL,
    season_label VARCHAR(20) NOT NULL,
    scraped_at   TIMESTAMPTZ,
    CONSTRAINT uq_season UNIQUE (league_id, season_start)
);

CREATE TABLE IF NOT EXISTS match_stats (
    id             BIGSERIAL    PRIMARY KEY,
    season_id      INT          NOT NULL REFERENCES seasons(id) ON DELETE CASCADE,
    match_date     VARCHAR(50),
    round_num      INT,
    match_name     VARCHAR(200) NOT NULL,
    score          VARCHAR(20),
    status         VARCHAR(50),
    url            VARCHAR(500),
    team           VARCHAR(100) NOT NULL,
    side           VARCHAR(10),
    opponent       VARCHAR(100),
    goals_scored   INT,
    goals_conceded INT,
    points         INT,
    stats          JSONB        NOT NULL DEFAULT '{}',
    ingested_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_match_stat UNIQUE (season_id, match_name, team)
);

CREATE INDEX IF NOT EXISTS idx_match_stats_season   ON match_stats (season_id);
CREATE INDEX IF NOT EXISTS idx_match_stats_match_name ON match_stats (match_name);
CREATE INDEX IF NOT EXISTS idx_match_stats_team     ON match_stats (team);
CREATE INDEX IF NOT EXISTS idx_match_stats_opponent ON match_stats (opponent);
CREATE INDEX IF NOT EXISTS idx_match_stats_stats    ON match_stats USING gin (stats);

-- ──────────────────────────────────────────────────────────────────────────────
-- League-level player & team ranking stats (FotMob stats pages)
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS player_season_stats (
    id               BIGSERIAL    PRIMARY KEY,
    season_id        INT          NOT NULL REFERENCES seasons(id) ON DELETE CASCADE,
    fotmob_season_id INT          NOT NULL,
    stat_category    VARCHAR(100) NOT NULL,
    rank             SMALLINT,
    player_fotmob_id BIGINT       NOT NULL,
    player_name      VARCHAR(200) NOT NULL,
    team_fotmob_id   BIGINT,
    team_name        VARCHAR(100),
    value            NUMERIC(12,3),
    ingested_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_player_season_stat UNIQUE (season_id, stat_category, player_fotmob_id)
);

CREATE INDEX IF NOT EXISTS idx_pss_season    ON player_season_stats (season_id);
CREATE INDEX IF NOT EXISTS idx_pss_category  ON player_season_stats (stat_category);
CREATE INDEX IF NOT EXISTS idx_pss_player    ON player_season_stats (player_fotmob_id);
CREATE INDEX IF NOT EXISTS idx_pss_team      ON player_season_stats (team_fotmob_id);

CREATE TABLE IF NOT EXISTS team_season_stats (
    id               BIGSERIAL    PRIMARY KEY,
    season_id        INT          NOT NULL REFERENCES seasons(id) ON DELETE CASCADE,
    fotmob_season_id INT          NOT NULL,
    stat_category    VARCHAR(100) NOT NULL,
    rank             SMALLINT,
    team_fotmob_id   BIGINT       NOT NULL,
    team_name        VARCHAR(200) NOT NULL,
    value            NUMERIC(12,3),
    ingested_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_team_season_stat UNIQUE (season_id, stat_category, team_fotmob_id)
);

CREATE INDEX IF NOT EXISTS idx_tss_season    ON team_season_stats (season_id);
CREATE INDEX IF NOT EXISTS idx_tss_category  ON team_season_stats (stat_category);
CREATE INDEX IF NOT EXISTS idx_tss_team      ON team_season_stats (team_fotmob_id);
