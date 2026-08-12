# Foreign Players Pipeline — Rollout (PR6)

## Stages

### Stage 1 — Shadow mode
Classify what *would* be persisted without writing.

```bash
# season_refresh
FOREIGN_SHADOW_MODE=1 python -m scripts.season_refresh ...

# dedicated backfill (last 2 seasons)
FOREIGN_SHADOW_MODE=1 python -m scripts.backfill_foreign_stats --seasons 2 --json
# or
python -m scripts.backfill_foreign_stats --seasons 2 --shadow
```

Inspect `would_persist`, `uncatalogued`, `persistence_rate`.

### Stage 2 — Persistence enabled (warning-only)
```bash
# Ensure migration 024 applied (comp_id nullable)
# Take DB backup first
python -m scripts.backfill_foreign_stats --baseline
python -m scripts.backfill_foreign_stats --seasons 2
python -m scripts.backfill_foreign_stats --baseline
```

Rate below 90% logs a **warning** but does not fail the Action.

### Stage 3 — ML fallback validation
Confirm path:
`player_season_stats` → `player_latest_stats_any_league` → loader →
`is_foreign_fallback=True` → Trainer inference-only prediction.

No training-set change.

### Stage 4 — Enforcement
```bash
FOREIGN_PERSISTENCE_ENFORCE=1
# optional thresholds:
FOREIGN_PERSISTENCE_RATE_OK=0.95
FOREIGN_PERSISTENCE_RATE_WARN=0.90
```

When enforce is on, `persistence_rate < WARN` → hard failure of
`season_refresh` / backfill.

## Env reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `FOREIGN_SHADOW_MODE` | off | Classify only, no DB writes |
| `FOREIGN_PERSISTENCE_ENFORCE` | off | Rate failure → hard error |
| `FOREIGN_PERSISTENCE_RATE_OK` | 0.95 | ≥ → ok |
| `FOREIGN_PERSISTENCE_RATE_WARN` | 0.90 | ≥ → warning; below → failure |

## Idempotency
Re-running backfill is safe: upserts on `leagues.name`,
`uq_season`, `uq_player_season_stat`.

## Safety checklist
1. DB backup / recovery point
2. `--baseline` counts before
3. `--shadow` dry-run
4. Live backfill
5. `--baseline` counts after — expect non-negative deltas, no 2× growth
