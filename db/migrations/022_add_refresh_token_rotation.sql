-- Migration: refresh token rotation + reuse detection + audit trail.
--
-- Problema risolto:
--   1. I refresh token non venivano mai eliminati dalla tabella (crescita
--      illimitata) — vedi 023_... per lo script di cleanup schedulato.
--   2. Non c'era rotation: uno stesso refresh token restava valido e
--      riusabile fino a scadenza naturale (fino a 30gg). Se rubato, un
--      attaccante poteva usarlo ripetutamente senza essere rilevato.
--
-- Soluzione: ogni token appartiene a una "family" (catena di rotation).
-- Ad ogni /refresh il token usato viene revocato e ne viene emesso uno
-- nuovo nella stessa family. Se un token già revocato viene ripresentato,
-- è la prova che qualcuno sta riusando un token rubato/obsoleto: revochiamo
-- l'intera family, forzando il logout su tutti i device compromessi.
--
-- Apply:
--   type db\migrations\022_add_refresh_token_rotation.sql | docker compose exec -T db psql -U fbref -d fbref

ALTER TABLE refresh_tokens
    ADD COLUMN IF NOT EXISTS family_id UUID NOT NULL DEFAULT gen_random_uuid();

ALTER TABLE refresh_tokens
    ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ NULL;

ALTER TABLE refresh_tokens
    ADD COLUMN IF NOT EXISTS revoked_reason VARCHAR(20) NULL
    CHECK (revoked_reason IN ('rotated', 'reuse_detected', 'logout', 'logout_all'));

COMMENT ON COLUMN refresh_tokens.family_id IS
    'Lega tutti i token emessi da una stessa catena di rotation. Un reuse su un token già revocato provoca la revoca dell''intera family.';
COMMENT ON COLUMN refresh_tokens.revoked_reason IS
    'rotated = sostituito da /refresh; reuse_detected = token rubato/obsoleto ripresentato; logout = revoca esplicita singola; logout_all = revoca di tutte le sessioni utente.';

CREATE INDEX IF NOT EXISTS idx_rt_family ON refresh_tokens (family_id);

-- Retention: i token già scaduti/revocati diventano candidati alla pulizia
-- (script api/scripts/cleanup_refresh_tokens.py), ma teniamo una finestra
-- di qualche giorno per indagini di sicurezza post-incidente.
CREATE INDEX IF NOT EXISTS idx_rt_cleanup ON refresh_tokens (revoked_at)
    WHERE revoked_at IS NOT NULL;
