-- Migration: add resolved_from_history column to player_id_map
--
-- Motivation: when the matching pipeline applies a historical manual
-- resolution (Pass 0), we want to mark the resulting player_id_map row
-- so the UI can display a "From history" badge and operators can audit
-- which mappings came from the permanent history vs. fresh automatic
-- matching.
--
-- This is a separate migration from 013 because it modifies the existing
-- player_id_map table rather than creating a new one.
--
-- Apply:
--   docker compose exec -T db psql -U fbref -d fbref \
--       -f /docker-entrypoint-initdb.d/014_add_resolved_from_history.sql

ALTER TABLE player_id_map
    ADD COLUMN IF NOT EXISTS resolved_from_history BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN player_id_map.resolved_from_history IS
    'TRUE when this mapping was applied from the manual_resolutions history (Pass 0).';
