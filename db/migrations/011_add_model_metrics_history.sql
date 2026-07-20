-- Migration 011: model run metrics history
--
-- Stores ML pipeline run metadata and per-metric results in Postgres so that
-- model performance can be queried over time, compared between runs, and
-- drift detected without parsing JSON artifacts on disk/R2.
--
-- Apply:
--   type db\migrations\011_add_model_metrics_history.sql | docker compose exec -T db psql -U fbref -d fbref
-- Or directly:
--   psql $DATABASE_URL -f db/migrations/011_add_model_metrics_history.sql

CREATE TABLE IF NOT EXISTS model_runs (
    id               SERIAL      PRIMARY KEY,
    run_id           TEXT        NOT NULL UNIQUE,   -- e.g. "20240101_120000" from trainer
    model_name       TEXT        NOT NULL,           -- "xgboost", "ridge", "random_forest", …
    trained_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season_start     INTEGER,                        -- latest season used (e.g. 2024)
    training_seasons JSONB,                          -- list of all seasons in training window
    hyperparams      JSONB,                          -- best model hyperparams (grid/Optuna)
    dependencies     JSONB,                          -- dep_versions dict collected by trainer
    git_commit       TEXT,                           -- git rev-parse --short HEAD (optional)
    -- 'ok' | 'degraded' | 'error'
    status           TEXT        NOT NULL DEFAULT 'ok'
);

-- Per-metric rows: one row per (run, metric, split) triple.
-- metric_name: 'rmse', 'mae', 'r2', …
-- split: 'test', 'train', 'backtest', 'gk', 'outfield', …
CREATE TABLE IF NOT EXISTS model_metrics (
    id           SERIAL          PRIMARY KEY,
    run_id       TEXT            NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    metric_name  TEXT            NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    split        TEXT            NOT NULL,
    n_samples    INTEGER
);

-- Alert row written when a run's RMSE degrades beyond threshold vs. moving avg.
CREATE TABLE IF NOT EXISTS model_drift_alerts (
    id             SERIAL          PRIMARY KEY,
    run_id         TEXT            NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    metric_name    TEXT            NOT NULL,
    current_value  DOUBLE PRECISION NOT NULL,
    baseline_value DOUBLE PRECISION NOT NULL,  -- moving avg of last N runs
    pct_change     DOUBLE PRECISION NOT NULL,   -- positive = degradation
    threshold_pct  DOUBLE PRECISION NOT NULL,
    created_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Indexes supporting time-series and per-model queries
CREATE INDEX IF NOT EXISTS idx_model_runs_trained_at    ON model_runs (trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_runs_model_name    ON model_runs (model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_run_metric ON model_metrics (run_id, metric_name, split);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_run         ON model_drift_alerts (run_id);
