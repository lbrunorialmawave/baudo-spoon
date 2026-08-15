# Changelog — Limited-cohort production hardening (2026-08-15)

## P0

### Configuration contract (WS1)
- Added `docs/config/auction-reliability-contract.md` as single source of truth.
- Defaults: `apply_reliability_weight=True`, `risk_aversion=0.0`, `reliability_weight_mode="continuous"`.
- Removed production-critical `False` fallbacks in:
  - `api/src/routers/auction.py`
  - `ml/auction/orchestrator.py`
- Constants exposed via `ml.auction.decision_score`.

### ADR cleanup (WS2)
- Canonical ADR: `docs/adr/0001-auction-reliability-weight-scope.md` (Option B).
- Conflicting draft renamed to `0001-limited-cohort-reliability-scope-HISTORICAL.md`.

### SHADOW semantics (WS3)
- `.github/workflows/ml-training.yml` no longer maps SHADOW → `ML_*=true`.
- Mapping:
  - ACTIVE → `ML_<FLAG>=true` (production path)
  - SHADOW → `ML_<FLAG>_CHALLENGER=true` (observe only)
  - DISABLED → no env

### Typed reliability_weight_mode (WS4)
- `MLConfig` validates mode ∈ {`bucket`, `continuous`}; rejects `true`/`false`/`foo`.
- Workflow supports string mode env vars (not bool-coerced).

## P1

### Decision policy (WS6)
- New `ml/auction/decision_score.py` with `compute_decision_score` / `compute_decision_score_from_player`.
- `VarEngine`, `alternatives.py`, and `simulation.py` use the canonical policy.
- Alternatives ranking no longer bypasses reliability weight.

### Promotion gate (WS5)
- `ml/scripts/check_promotion_gate.py`:
  - Fail-closed on missing required metrics.
  - Hard canary gate: `canary_anomalies_remaining > 0` → DENY.
  - Over-representation delta support.
  - Exit 0 / 1 / 2 as specified.
  - `--json-summary` machine-readable output.

### Tests
- `ml/tests/test_production_hardening_invariants.py` (WS9 invariants, decision score, gate, workflow static check).
- Existing `test_promotion_gate.py` updated for canary fields.

### Dependency preflight (WS0.3)
- `ml/scripts/check_test_dependencies.py` (explicit error + install hint for `pulp` etc.).

### Runbook (WS13)
- `docs/runbooks/limited-cohort-rollout.md`.

## Remaining (not yet closed)

- Full E2E contract tests (Trainer → Optimizer → Auction → alternatives → simulation → rollout stages).
- Frontend Angular contract/round-trip automated tests (defaults already correct in component).
- Automated rollback integration test.
- Observability metrics export (WS12).
- Trainer consumption of `*_CHALLENGER` env vars for shadow artifact emission.
- CI ADR uniqueness guard.

## Batch 2 — 2026-08-15 (continued)

### Trainer challenger path (WS3)
- `ml/rollout/env_flags.py`: resolves `ML_*=true` (ACTIVE) vs `ML_*_CHALLENGER=true` (SHADOW).
- `ml/rollout/shadow_artifacts.py`: baseline vs challenger decision-score comparison artifact.
- `Trainer._maybe_write_shadow_artifact`: writes `shadow_comparison_<run_id>.json` when challenger flags are set, without changing production decisions.

### E2E decision contract (WS8)
- `ml/tests/test_e2e_decision_contract.py`: golden LIMITED player aligned across decision_score, VarEngine, alternatives, simulation; rollout stages; env flag resolution; shadow artifact.

### Rollback (WS11)
- `ml/tests/test_rollback_and_adr_guard.py`: ACTIVE→DISABLED, continuous→bucket, apply_reliability_weight kill-switch, gate-failure blocks promotion.

### ADR CI guard (WS2)
- `ml/scripts/check_adr_uniqueness.py` + unit tests.

## Batch 3 — 2026-08-15 (final)

### Observability (WS12)
- `ml/rollout/observability.py`: cohort counts, overrepresentation, mean weights/minutes, score layers (raw/display/decision).
- `ml/tests/test_observability.py`

### Frontend contract (WS7)
- `ml/tests/test_frontend_auction_contract.py`: static TS checks for defaults, setupAuctionConfig, startAuction, boolean hydration.

### CI hardening
- `.github/workflows/ci.yml`: dependency preflight + ADR uniqueness before ML pytest.

**Test suite (hardening-related): 72 passed.**
