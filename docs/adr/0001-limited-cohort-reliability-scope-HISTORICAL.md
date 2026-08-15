# ADR 0001 — LIMITED cohort reliability scope (Optimizer vs Auction)

- **Status:** Accepted (technical recommendation; product defaults remain conservative)
- **Date:** 2026-08-15
- **Context:** plan-limited-cohort-hardening.md WS3
- **Deciders:** Engineering (implementation), product owner (activation thresholds)

## Context

Players in the LIMITED cohort (100–799 minutes) can produce statistically extreme per-90 features. The system already applies:

1. **Input shrinkage** (flag-gated: `enable_shrinkage`) on training features.
2. **Output / display shrinkage** (`attach_output_reliability`) on predictions shown downstream.
3. **Decision reliability weight** in the Optimizer ILP objective (`Player.reliability_weight`).

Historically, Auction `VarEngine` used only the already-shrunk `projected_score` and did **not** apply `reliability_weight` or `prediction_std` penalties. That asymmetry was never an explicit product decision.

## Decision

**Option B — shrink + decision-weight everywhere (opt-in for Auction):**

| Layer | Optimizer | Auction (`VarEngine`) |
|-------|-----------|------------------------|
| Display shrinkage | via data_repository / artifact | same |
| `reliability_weight` multiplier | always (when present on Player) | **opt-in** via `AuctionConfig.apply_reliability_weight` (default `False`) |
| `risk_aversion * prediction_std` | `OptimizationConfig.risk_aversion` | **opt-in** via `AuctionConfig.risk_aversion` (default `0.0`) |

Rationale:

- Output shrinkage is a **presentation** correction.
- Decision weight + risk aversion are **automatic selection/ranking** corrections.
- Keeping Auction defaults off preserves bit-identical live auction behaviour until operators explicitly enable alignment.
- Optimizer-side VAR enrichment (`/optimizer` pool enrichment) enables `apply_reliability_weight=True` so VAR/ESV blends used inside the ILP stay consistent with the ILP reliability discount.

## Consequences

- Positive: single conceptual model; continuous reliability weight (WS2) can flow to both modules once enabled.
- Positive: kill-switches remain independent (`apply_reliability_weight`, `risk_aversion`).
- Negative: until Auction flags are turned on, LIMITED ranking risk remains higher in Auction than Optimizer.
- Follow-up: after canary + harness gate (WS4) pass with `enable_shrinkage` ACTIVE, recommend enabling `apply_reliability_weight=True` and a small `risk_aversion` (e.g. 0.25–0.5) in production auction presets.

## Alternatives considered

- **Option A (shrink-once):** only `projected_score` carries damping. Rejected as primary path because display shrinkage is not calibrated for selection severity.
- Raising `min_minutes_hard`: rejected for this phase (throws away legitimate emerging-player signal).
