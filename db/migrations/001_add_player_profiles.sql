-- Migration: add player_profiles table for role/position data.
-- Backward-compatible: does not alter existing tables.
-- Apply once: psql -U <user> -d <db> -f 001_add_player_profiles.sql

CREATE TABLE IF NOT EXISTS player_profiles (
    player_fotmob_id  BIGINT       PRIMARY KEY,
    player_name       VARCHAR(200) NOT NULL,
    role_key          VARCHAR(50),                       -- raw FotMob position key
    canonical_role    VARCHAR(5)
        CHECK (canonical_role IN ('GK', 'DEF', 'MID', 'FWD')),
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pp_canonical_role
    ON player_profiles (canonical_role);
