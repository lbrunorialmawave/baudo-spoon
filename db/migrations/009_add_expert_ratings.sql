-- Migration: add expert_ratings table for third-party expert opinions.
--
-- Stores ratings and comments from expert sources (e.g. forum.gruppoesperti.it).
-- These are purely informational overlays — they do NOT affect MANTRA scores
-- (FP, VR, Prezzo). Displayed alongside algorithmic evaluations in the UI.

CREATE TABLE IF NOT EXISTS expert_ratings (
    id               BIGSERIAL    PRIMARY KEY,
    player_id        VARCHAR(50)  NOT NULL,          -- "fc-{fantacalcio_id}" or fotmob_id
    source           VARCHAR(50)  NOT NULL,           -- e.g. "gruppo_esperti"
    expert_name      VARCHAR(100),                    -- e.g. "baghino", "terzino90"
    rating           SMALLINT,                        -- 1-5 stelle (NULL if comment-only)
    comment          TEXT,                            -- free-text comment
    matchday         INT,                             -- reference matchday (NULL = pre-season)
    season_start     INT          NOT NULL,
    url              VARCHAR(500),                    -- link to original post
    scraped_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_expert_rating UNIQUE (player_id, source, expert_name, matchday)
);

COMMENT ON TABLE  expert_ratings IS
    'Third-party expert ratings and comments. Informational only — does not affect MANTRA scores.';
COMMENT ON COLUMN expert_ratings.rating IS
    'Star rating 1-5 (NULL if only a textual comment).';

CREATE INDEX IF NOT EXISTS idx_er_player   ON expert_ratings (player_id);
CREATE INDEX IF NOT EXISTS idx_er_source   ON expert_ratings (source);
CREATE INDEX IF NOT EXISTS idx_er_season   ON expert_ratings (season_start);
