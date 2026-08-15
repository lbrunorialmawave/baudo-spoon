# Limited-cohort Rollout Runbook

**Date:** 2026-08-15  
**Related:** ADR 0001, `docs/config/auction-reliability-contract.md`, production-hardening plan

## 1. Prerequisites

- Full test suite green (including dependency preflight)
- Canary fixture available (`ml/tests/fixtures/limited_cohort_canary.py`)
- ADR 0001 canonical present
- Config contract present
- No open P0/P1 blockers

## 2. Baseline

```bash
python -m ml.scripts.check_test_dependencies
pytest ml/tests/test_limited_cohort_hardening.py -q
pytest ml/tests/test_sample_reliability.py -q
pytest ml/tests/test_rollout.py -q
pytest ml/tests/test_promotion_gate.py -q
```

Record SHA, Python/Node versions, lockfile hashes.

## 3. SHADOW

1. Promote flag(s) to SHADOW via `ml-rollout` workflow (action=shadow).
2. Run training / experiment pipeline.
3. Confirm production decision path is still legacy (no `ML_<FLAG>=true` for production use).
4. Collect challenger artifacts: baseline vs challenger predictions, delta, cohort, canary status.

## 4. Metrics

Required in every experiment report:

- mae / rmse (aggregate + by cohort)
- phenom_leakage_rate / phenom_overrepresentation
- canary_anomalies_total / resolved / remaining

## 5. Promotion gate

```bash
python -m ml.scripts.check_promotion_gate --report <path>
```

Must exit 0. Fail closed on missing metrics or canary_anomalies_remaining > 0.

## 6. ACTIVE

Only after:

- SHADOW report accepted
- canary anomalies = 0
- leakage within threshold
- aggregate regression within threshold
- config drift = 0

Use `ml-rollout` action=activate.

## 7. Rollback

- ACTIVE → DISABLED (or SHADOW) restores legacy decision path.
- `apply_reliability_weight=True` → `False` without schema change.
- continuous → bucket via mode flag.

Automated rollback test must pass before any ACTIVE promotion.

## 8. Incident response

1. Kill-switch: set stage DISABLED via rollout workflow.
2. Verify production path reverted.
3. Capture artifacts and canary status.
4. Open incident with report fields above.

## 9–11. Verification checklist (pre-ACTIVE)

- [ ] test suite completa verde
- [ ] dependency preflight verde
- [ ] canary disponibile
- [ ] canary anomalies = 0
- [ ] MAE/RMSE cohort report presente
- [ ] leakage entro soglia
- [ ] config drift = 0
- [ ] ADR canonica presente
- [ ] frontend contract test verde
- [ ] E2E Auction verde
- [ ] E2E Optimizer verde
- [ ] rollback testato
