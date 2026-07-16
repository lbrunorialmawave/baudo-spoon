-- Migration: fix NUMERIC precision for qt_a_norm / qt_i_norm generated columns
--
-- Rationale: the original ``NUMERIC(6,4)`` precision allows values up to
-- 99.9999, which corresponds to a raw qt_a up to ~29999 (≈ 99.99 × 300).
-- Real Fantacalcio values rarely exceed 500 (FVM mode), but the margin
-- is too tight: a value of 999 would already produce 3.33, which fits,
-- but with any future FVM > 999 the insert would raise:
--   numeric field overflow
--   A field with precision 6, scale 4 must round to an absolute value < 1000.
--
-- The fix widens to ``NUMERIC(8,5)``: max value 999.99999, safe up to
-- qt_a = 299,999.  This is 100× the current max and costs negligible
-- storage (PostgreSQL numeric is variable-width).
--
-- PostgreSQL does not allow ALTER of a GENERATED column's expression or
-- type in-place.  We must DROP and re-CREATE.  The table is populated
-- by ``import_quotations`` (a batch job), so this is safe.
--
-- Apply once:
--   docker compose exec -T db psql -U fbref -d fbref \
--       -f /docker-entrypoint-initdb.d/005_fix_qt_norm_precision.sql

ALTER TABLE player_quotations
    DROP COLUMN IF EXISTS qt_a_norm,
    DROP COLUMN IF EXISTS qt_i_norm;

ALTER TABLE player_quotations
    ADD COLUMN qt_a_norm NUMERIC(8,5)
        GENERATED ALWAYS AS (qt_a / 300.0) STORED,
    ADD COLUMN qt_i_norm NUMERIC(8,5)
        GENERATED ALWAYS AS (qt_i / 300.0) STORED;
