# Manifest — Limited-cohort production hardening deliverables

**Date:** 2026-08-15  
**Project:** baudo-spoon  
**Related plan:** plan-limited-cohort-production-hardening.md

## New files

| Path | Purpose |
|------|---------|
| `ml/auction/decision_score.py` | Canonical decision-score policy (WS6) |
| `ml/rollout/env_flags.py` | ACTIVE vs SHADOW env resolution (WS3) |
| `ml/rollout/shadow_artifacts.py` | Shadow comparison artifacts (WS3) |
| `ml/rollout/observability.py` | Aggregate PII-free metrics (WS12) |
| `ml/scripts/check_test_dependencies.py` | Dependency preflight (WS0.3) |
| `ml/scripts/check_adr_uniqueness.py` | ADR uniqueness CI guard (WS2) |
| `ml/tests/test_production_hardening_invariants.py` | WS9 invariants |
| `ml/tests/test_e2e_decision_contract.py` | WS8 E2E decision contract |
| `ml/tests/test_rollback_and_adr_guard.py` | WS11 rollback + ADR tests |
| `ml/tests/test_frontend_auction_contract.py` | WS7 frontend static contract |
| `ml/tests/test_observability.py` | WS12 tests |
| `docs/config/auction-reliability-contract.md` | Config contract (WS1) |
| `docs/runbooks/limited-cohort-rollout.md` | Production runbook (WS13) |
| `docs/CHANGELOG-production-hardening-2026-08-15.md` | Technical changelog |
| `docs/MANIFEST-production-hardening.md` | This file |

## Modified files

| Path | Purpose |
|------|---------|
| `docs/adr/0001-auction-reliability-weight-scope.md` | Canonical Option B ADR |
| `docs/adr/0001-limited-cohort-reliability-scope-HISTORICAL.md` | Superseded draft |
| `.github/workflows/ml-training.yml` | SHADOW ≠ ACTIVE env mapping |
| `.github/workflows/ci.yml` | Preflight + ADR guard steps |
| `ml/config.py` | `reliability_weight_mode` validator |
| `ml/auction/alternatives.py` | Decision-score ranking |
| `ml/auction/simulation.py` | Decision-score ESV proxy |
| `ml/auction/var.py` | Canonical decision score |
| `ml/auction/orchestrator.py` | Default apply_reliability_weight=True |
| `ml/pipeline/trainer.py` | Env production flags + shadow artifact |
| `ml/rollout/__init__.py` | Public exports |
| `ml/scripts/check_promotion_gate.py` | Canary + fail-closed (WS5) |
| `ml/tests/test_promotion_gate.py` | Canary fields |
| `api/src/routers/auction.py` | Default True fallbacks |

## Archives (workspace)

| File | Description |
|------|-------------|
| `/home/workdir/artifacts/baudo-spoon-production-hardening.zip` | Full project snapshot |
| `/home/workdir/artifacts/hardening-files-only.tgz` | Hardening files only |

## Verification

```bash
cd baudo-spoon
python ml/scripts/check_adr_uniqueness.py
python -m ml.scripts.check_test_dependencies   # requires full deps
ML_DATABASE_URL=postgresql://x:x@localhost/x \
  pytest ml/tests/test_production_hardening_invariants.py \
         ml/tests/test_e2e_decision_contract.py \
         ml/tests/test_rollback_and_adr_guard.py \
         ml/tests/test_frontend_auction_contract.py \
         ml/tests/test_observability.py \
         ml/tests/test_promotion_gate.py -q
```
