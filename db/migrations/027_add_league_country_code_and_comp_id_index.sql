-- Migration 027: add country_code to leagues; partial unique index on comp_id
--
-- Phase 3 of the league identity fix (plan §2.3 / Issue #2):
--   leagues.name is NOT a stable identity when display names collide across
--   countries (e.g. "Serie A" is both ITA comp_id=55 and BRA comp_id=268).
--
-- This migration:
--   1. Adds leagues.country_code (nullable) for observability and future queries.
--   2. Adds a partial UNIQUE index on comp_id WHERE NOT NULL so catalogued
--      leagues cannot be double-inserted even while the name-based upsert key
--      is still in use (Phase 5 will retire the name key).
--   3. Backfills country_code for the five catalogued leagues already in the DB.
--
-- Pending work (Phase 4-5, separate migrations):
--   - Update get_or_create_league() to use comp_id as the upsert conflict key.
--   - Drop UNIQUE(name); add UNIQUE(comp_id) as the primary identity constraint.
--   - Migrate WHERE l.name='Serie A' queries to WHERE l.comp_id='55'.
--
-- Apply:
--   psql -d defaultdb -f db/migrations/027_add_league_country_code_and_comp_id_index.sql
-- Or via docker-compose:
--   type db/migrations/027_add_league_country_code_and_comp_id_index.sql | docker compose exec -T db psql -U fbref -d fbref

-- 1. Add country_code column (nullable until fully backfilled).
ALTER TABLE leagues
    ADD COLUMN IF NOT EXISTS country_code VARCHAR(3);

-- 2. Partial unique index: comp_id must be unique among catalogued rows.
--    NULL rows (uncatalogued foreign-career leagues) are intentionally excluded.
CREATE UNIQUE INDEX IF NOT EXISTS uq_leagues_comp_id
    ON leagues (comp_id)
    WHERE comp_id IS NOT NULL;

-- 3. Backfill country_code for all currently catalogued leagues.
UPDATE leagues SET country_code = 'ITA' WHERE comp_id IN ('55', '157');
UPDATE leagues SET country_code = 'ENG' WHERE comp_id = '47';
UPDATE leagues SET country_code = 'ESP' WHERE comp_id = '87';
UPDATE leagues SET country_code = 'GER' WHERE comp_id = '54';
UPDATE leagues SET country_code = 'FRA' WHERE comp_id = '53';
