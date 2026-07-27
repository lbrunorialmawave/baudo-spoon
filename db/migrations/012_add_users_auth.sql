-- Migration: add users and refresh_tokens tables for JWT-based auth.
--
-- Replaces the single shared API key for human users.
-- Machine-to-machine (scraper cron) continues using the API key via X-API-Key.
--
-- Apply:
--   type db\migrations\012_add_users_auth.sql | docker compose exec -T db psql -U fbref -d fbref

CREATE TABLE IF NOT EXISTS users (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(20)  NOT NULL CHECK (role IN ('admin', 'member')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  users IS 'Application users. Roles: admin (full access), member (read + optimizer/auction).';
COMMENT ON COLUMN users.role IS 'admin or member — enforced by require_role() FastAPI dependency.';

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  VARCHAR(255) UNIQUE NOT NULL,
    expires_at  TIMESTAMPTZ  NOT NULL,
    revoked     BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  refresh_tokens IS 'Opaque refresh tokens stored hashed. Revocation via revoked flag.';

CREATE INDEX IF NOT EXISTS idx_rt_user    ON refresh_tokens (user_id);
CREATE INDEX IF NOT EXISTS idx_rt_hash    ON refresh_tokens (token_hash);
CREATE INDEX IF NOT EXISTS idx_rt_expiry  ON refresh_tokens (expires_at) WHERE NOT revoked;
