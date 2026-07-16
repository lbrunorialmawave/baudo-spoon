-- Migration: add player_quotations and player_id_map tables for Fantacalcio
-- market data integration.
--
-- Two tables:
--   1. player_quotations  — one row per (fantacalcio_id, season_start).
--      Holds the raw Qt.A / Qt.I (current and initial auction value) plus
--      Mantra variants and FVM (fantavalori medi) for cross-validation.
--   2. player_id_map      — deterministic link between Fantacalcio IDs
--      (used by the .xlsx listoni) and player_fotmob_id (used by the rest
--      of the pipeline). One row per (fantacalcio_id, season_start) so
--      that the same Fantacalcio id can be re-assigned across years if
--      the operator notices a mis-match.
--
-- Backward-compatible: does not alter existing tables.
-- Apply once: psql -U <user> -d <db> -f 002_add_player_quotations.sql

-- ────────────────────────────────────────────────────────────────────────────
-- 1. player_quotations
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS player_quotations (
    id               BIGSERIAL   PRIMARY KEY,
    fantacalcio_id   INT         NOT NULL,
    season_start     INT         NOT NULL,
    role             VARCHAR(5)  NOT NULL
        CHECK (role IN ('GK', 'DEF', 'MID', 'FWD')),
    team             VARCHAR(100) NOT NULL,
    player_name      VARCHAR(200) NOT NULL,

    -- Core Fantacalcio values (Classic mode)
    qt_a             INT         NOT NULL,    -- valutazione attuale
    qt_i             INT         NOT NULL,    -- valutazione iniziale (asta)
    diff_val         INT         NOT NULL,    -- qt_a - qt_i (variazione)

    -- Mantra-mode variants (optional, may be NULL when listone doesn't split)
    qt_a_m           INT,
    qt_i_m           INT,
    diff_val_m       INT,

    -- FVM (fantavalori medi) — internal Fantacalcio consensus, useful for
    -- cross-validation against the ML target.
    fvm              INT,
    fvm_m            INT,

    -- Normalised variants (qt / 300) for cross-season comparability.
    -- Computed on insert via a generated column so downstream queries are
    -- trivial and never go out of sync with the raw value.
    qt_a_norm        NUMERIC(6,4) GENERATED ALWAYS AS (qt_a / 300.0) STORED,
    qt_i_norm        NUMERIC(6,4) GENERATED ALWAYS AS (qt_i / 300.0) STORED,

    source           VARCHAR(50) NOT NULL DEFAULT 'listone_fantagazzetta',
    imported_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_player_quotation UNIQUE (fantacalcio_id, season_start)
);

CREATE INDEX IF NOT EXISTS idx_pq_season      ON player_quotations (season_start);
CREATE INDEX IF NOT EXISTS idx_pq_role        ON player_quotations (role);
CREATE INDEX IF NOT EXISTS idx_pq_team        ON player_quotations (team);
CREATE INDEX IF NOT EXISTS idx_pq_season_role ON player_quotations (season_start, role);

-- ────────────────────────────────────────────────────────────────────────────
-- 2. player_id_map — Fantacalcio ID ↔ FotMob ID
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS player_id_map (
    id                 BIGSERIAL   PRIMARY KEY,
    fantacalcio_id     INT         NOT NULL,
    season_start       INT         NOT NULL,
    player_fotmob_id   BIGINT,                    -- nullable: unmatched rows
    name_fantacalcio   VARCHAR(200) NOT NULL,
    name_fotmob        VARCHAR(200),
    team_fantacalcio   VARCHAR(100),
    team_fotmob        VARCHAR(100),
    canonical_role     VARCHAR(5)
        CHECK (canonical_role IN ('GK', 'DEF', 'MID', 'FWD')),
    match_method       VARCHAR(30) NOT NULL
        CHECK (match_method IN (
            'exact_name_team',
            'exact_name_role',
            'fuzzy_name',
            'manual',
            'unmatched'
        )),
    confidence         NUMERIC(4,3) NOT NULL DEFAULT 1.0
        CHECK (confidence BETWEEN 0 AND 1),
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_id_map UNIQUE (fantacalcio_id, season_start)
);

CREATE INDEX IF NOT EXISTS idx_pim_fotmob    ON player_id_map (player_fotmob_id)
    WHERE player_fotmob_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pim_season    ON player_id_map (season_start);
CREATE INDEX IF NOT EXISTS idx_pim_method    ON player_id_map (match_method);
CREATE INDEX IF NOT EXISTS idx_pim_role      ON player_id_map (canonical_role);
