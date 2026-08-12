# Foreign Players Pipeline — Rollout (PR7, season-aware)

Season-aware foreign pipeline (PR1–PR7). Guiding rules:

1. `LEAGUE_CATALOG` decides bulk scrape scope, **not** what may be persisted.
2. `target_season_start` decides which season to seek; a newer FotMob season
   must never silently replace the target.
3. Historical prediction uses a target-aware DB path
   (`player_stats_by_prediction_season`); `player_latest_stats_any_league`
   remains the absolute-latest consumer only.

## Migrations required

| Migration | Purpose |
|-----------|---------|
| 024 | `leagues.comp_id` nullable (uncatalogued leagues) |
| 025 | `source_season_start`, `prediction_season_start`, reason, depth |
| 026 | target-aware view `player_stats_by_prediction_season` |

## Stages

### Stage 0 — Baseline

```bash
python -m scripts.backfill_foreign_stats --baseline --json
```

Record: leagues, seasons, player_season_stats, latest view, uncatalogued,
lineage coverage, target-aware view row count.

### Stage 1 — Shadow (no DB writes)

```bash
FOREIGN_SHADOW_MODE=1 python -m scripts.backfill_foreign_stats --seasons 2 --json
# or
python -m scripts.backfill_foreign_stats --seasons 2 --shadow --json
```

Inspect:

- `would_persist`, `uncatalogued`, `persistence_rate`
- season-resolution: `season_target_selected`, `season_previous_selected`,
  `season_no_valid`, `season_fallback_depth_histogram`
- conservation invariants (`invariant_ok`)

DB must be unchanged vs Stage 0 baseline.

### Stage 2 — One-season persist (warning-only)

```bash
# DB backup first
python -m scripts.backfill_foreign_stats --baseline
python -m scripts.backfill_foreign_stats --seasons 1
python -m scripts.backfill_foreign_stats --baseline
python -m scripts.backfill_foreign_stats --health
```

Rate below 90% → warning only (unless `FOREIGN_PERSISTENCE_ENFORCE=1`).

### Stage 3 — Two-season persist

```bash
python -m scripts.backfill_foreign_stats --seasons 2
python -m scripts.backfill_foreign_stats --health --json
```

### Stage 4 — ML path validation

Confirm:

```text
player_season_stats (lineage)
  → player_stats_by_prediction_season   # target-aware
  → ml/data/loader._append_foreign_fallback_rows
  → is_foreign_fallback=True
  → trainer quarantine (no train / no eval)
  → inference prediction
```

Absolute-latest view remains available for non-historical consumers.

### Stage 5 — Enforcement

```bash
FOREIGN_PERSISTENCE_ENFORCE=1
FOREIGN_PERSISTENCE_RATE_WARN=0.90
FOREIGN_PERSISTENCE_RATE_OK=0.95
python -m scripts.backfill_foreign_stats --seasons 1
```

When enforce is on:

- `persistence_rate < WARN` → hard failure
- failed health gates → hard failure

## Env reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `FOREIGN_SHADOW_MODE` | off | Classify only, no DB writes |
| `FOREIGN_PERSISTENCE_ENFORCE` | off | Rate / health failure → hard error |
| `FOREIGN_PERSISTENCE_RATE_OK` | 0.95 | ≥ → ok |
| `FOREIGN_PERSISTENCE_RATE_WARN` | 0.90 | ≥ → warning; below → failure if enforce |

## Idempotency

Re-running backfill is safe: upserts on `leagues.name`, `uq_season`,
`uq_player_season_stat`. Second run → 0 new logical rows.

Candidates are keyed by `(player_fotmob_id, target_season_start)` — never
collapsed to `dict[player_id] = name` in multi-season batches.

## Health gates (`--health`)

| Gate | Meaning |
|------|---------|
| `latest_view_readable` | `player_latest_stats_any_league` queryable |
| `target_aware_view_readable` | `player_stats_by_prediction_season` queryable |
| `lineage_columns_queryable` | migration 025 columns present |
| `foreign_sentinel_queryable` | `fotmob_season_id = -1` countable |
| `conservation_invariants` | candidates = fetched + unresolved (when result present) |
| `candidates_accounted` | same accounting check |
| `season_resolution_ran` | season metrics consistent with fetches |

## Rollback (plan §43)

1. Disable persistence (`FOREIGN_SHADOW_MODE=1` or stop workflow)
2. Stop backfill
3. Preserve metrics/logs
4. Rollback code if required
5. **Do not** blind `DELETE` rows
6. Analyze affected `source_season_start` / `prediction_season_start`
7. Corrective migration only after evidence

## Safety checklist

1. DB backup / recovery point
2. `--baseline` before
3. `--shadow` dry-run — review `would_persist` + season metrics
4. Live backfill (1 season, then 2)
5. `--baseline` after + `--health`
6. Confirm ML quarantine still excludes foreign from training
