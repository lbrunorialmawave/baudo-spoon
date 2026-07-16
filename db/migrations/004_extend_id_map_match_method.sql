-- Migration: extend player_id_map.match_method CHECK constraint
--
-- Rationale: the import_quotations pipeline was upgraded to add a
-- fourth matching strategy ("exact_relaxed_role") that ignores
-- canonical_role when joining on (surname, team). This recovers the
-- common case where Fantacalcio and FotMob classify the same player
-- differently (e.g. a winger quoted as MID in the listone but FWD on
-- FotMob). The original CHECK constraint only allows:
--   exact_name_team, exact_name_role, fuzzy_name, manual, unmatched
-- and rejects any new value with:
--   psycopg2.errors.CheckViolation:
--     new row for relation "player_id_map" violates check constraint
--     "player_id_map_match_method_check"
--
-- This migration drops the old constraint and re-creates it with the
-- extra value 'exact_relaxed_role' allowed. No data is touched: this
-- is a pure schema change.
--
-- Backward-compatible: safe to apply while the API is running — the
-- CHECK is only checked at row INSERT/UPDATE time, and existing rows
-- already use values from the original allowed set.
--
-- Apply once:
--   docker compose exec -T db psql -U fbref -d fbref \
--       -f /docker-entrypoint-initdb.d/004_extend_id_map_match_method.sql
-- (or run the file contents via the psql -c flag below if you prefer
--  to stay outside the container).

ALTER TABLE player_id_map
    DROP CONSTRAINT IF EXISTS player_id_map_match_method_check;

ALTER TABLE player_id_map
    ADD CONSTRAINT player_id_map_match_method_check
    CHECK (match_method IN (
        'exact_name_team',
        'exact_name_role',
        'exact_name_team_role_season',
        'exact_relaxed_role',
        'fuzzy_name',
        'manual',
        'unmatched'
    ));
