-- Migration: expose why an automatic player-id match needs review.
-- Used by season onboarding observability and the unresolved CSV export.

ALTER TABLE player_id_map
    ADD COLUMN IF NOT EXISTS reason TEXT;

COMMENT ON COLUMN player_id_map.reason IS
    'Optional operator-facing reason for a downgraded/unresolved match (e.g. team_mismatch, low_fuzzy_score, no_candidate).';
