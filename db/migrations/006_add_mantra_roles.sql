-- Migration: add player_mantra_roles table for 12-role MANTRA support.
--
-- Fantacalcio listoni contain a column ``rm`` with MANTRA role codes
-- (e.g. "Dd;E", "A", "T;W") that the original import (migration 002)
-- read but never persisted.  This table stores:
--
--   1. ruoli_mantra   — the full list of roles a player can cover (array)
--   2. ruolo_primario — the deepest (most defensive) role, computed via
--      the MANTRA depth hierarchy: Por→Dc,B,Dd,Ds→E,M→C→T,W→A,Pc
--
-- One row per (fantacalcio_id, season_start), same key as player_quotations.
-- The data is populated by the import_quotations CLI (modified accordingly).
-- The FK ensures referential integrity: no mantra role row without a
-- corresponding quotation row.
--
-- Apply: psql -U <user> -d <db> -f 006_add_mantra_roles.sql

CREATE TABLE IF NOT EXISTS player_mantra_roles (
    fantacalcio_id    INT         NOT NULL,
    season_start      INT         NOT NULL,
    ruolo_primario    VARCHAR(5)  NOT NULL,
    ruoli_mantra      TEXT[]      NOT NULL DEFAULT '{}',

    PRIMARY KEY (fantacalcio_id, season_start),
    FOREIGN KEY (fantacalcio_id, season_start)
        REFERENCES player_quotations (fantacalcio_id, season_start)
        ON DELETE CASCADE,

    CONSTRAINT chk_ruolo_primario
        CHECK (ruolo_primario IN (
            'Por', 'Dc', 'Dd', 'Ds', 'B',
            'E', 'M',
            'C',
            'T', 'W',
            'A', 'Pc'
        ))
);

COMMENT ON TABLE  player_mantra_roles IS
    'MANTRA 12-role system: maps each Fantacalcio player to their primary and secondary roles.';
COMMENT ON COLUMN player_mantra_roles.ruolo_primario IS
    'Most defensive / deepest role per MANTRA depth hierarchy (Por→Dc→E→C→T→A).';
COMMENT ON COLUMN player_mantra_roles.ruoli_mantra IS
    'All MANTRA roles the player can cover, from the listone rm column (e.g. {Dd,E}).';

CREATE INDEX IF NOT EXISTS idx_pmr_primario ON player_mantra_roles (ruolo_primario);
CREATE INDEX IF NOT EXISTS idx_pmr_season   ON player_mantra_roles (season_start);
