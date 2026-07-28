-- Migration: add 'fotmob_suggest' to player_id_map.match_method CHECK constraint
--
-- Rationale: the ID mapping pipeline was extended with a fourth pass that
-- calls FotMob's public /api/data/search/suggest API for players who remain
-- unmatched after exact + fuzzy matching. If exactly one result is returned
-- it is accepted automatically with match_method = 'fotmob_suggest'.
--
-- The current CHECK constraint (from 004) allows:
--   exact_name_team, exact_name_role, exact_name_team_role_season,
--   exact_relaxed_role, fuzzy_name, manual, unmatched
-- and rejects 'fotmob_suggest' with a CheckViolation.
--
-- This migration drops the old constraint and re-creates it with the
-- extra value included.
--
-- Backward-compatible: safe to apply while the API is running.
--
-- Apply via:
--   docker compose exec -T db psql -U fbref -d fbref \
--       -f /docker-entrypoint-initdb.d/015_add_fotmob_suggest_method.sql

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
        'fotmob_suggest',
        'manual',
        'unmatched'
    ));
