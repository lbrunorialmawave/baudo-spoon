# Auction / Optimizer Reliability Configuration Contract

**Owner:** ML Engineering  
**Date:** 2026-08-15  
**Status:** Canonical source of truth  
**Related ADR:** `docs/adr/0001-auction-reliability-weight-scope.md`

## Canonical fields

| Field | Type | Default | Valid values | Notes |
|-------|------|---------|--------------|-------|
| `apply_reliability_weight` | `bool` | `true` | `true`, `false` | Decision-layer multiplier on projected_score |
| `risk_aversion` | `float` | `0.0` | `[0.0, 5.0]` | Multiplier of `prediction_std` penalty; opt-in until calibrated |
| `reliability_weight_mode` | `str` | `"continuous"` | `"bucket"`, `"continuous"` | How minutes map to reliability weight |

## Ownership

- **Definition / defaults:** `ml.auction.models.AuctionConfig` and `api.src.schemas.AuctionConfigSchema` (must stay in sync). Single constants preferred in a shared config module.
- **Rollout control:** `ml.rollout.controller` stages map to effective values (see below).
- **Runtime consumers:** VarEngine, Optimizer solver, alternatives, simulation, data_repository hydration.

## Fallback / hydration rules

1. Schema validation first (Pydantic / TypeScript models).
2. Normalization step produces a canonical `AuctionConfig`.
3. Every consumer reads only the canonical object; no consumer may re-interpret a missing field as `False`.
4. Explicit `false` / `0.0` / `"bucket"` in a payload must be preserved (round-trip safe).

```text
raw payload
   ↓
schema validation
   ↓
normalization (defaults applied only for absent fields)
   ↓
canonical AuctionConfig
   ↓
all consumers (VarEngine, Optimizer, alternatives, simulation, frontend)
```

## Rollout semantics

| Stage | Production decision path | Challenger computation |
|-------|--------------------------|------------------------|
| DISABLED | legacy (no reliability weight / shrinkage off) | disabled |
| SHADOW | legacy | enabled; artifacts only |
| ACTIVE | challenger (after promotion gate) | enabled |

For `reliability_weight_mode`:

| Stage | Effective mode (production) | Challenger |
|-------|-----------------------------|------------|
| DISABLED | `bucket` (legacy) | — |
| SHADOW | `bucket` | may compute continuous |
| ACTIVE | `continuous` (or configured) | — |

## Backward compatibility

- Clients that omit the fields receive the canonical defaults above.
- Clients that send explicit `applyReliabilityWeight: false` keep that value.
- No database schema change.
- Kill-switch: set stage to DISABLED or pass `apply_reliability_weight=false`.

## Prohibited patterns

```python
# FORBIDDEN – diverging fallback
cfg.get("apply_reliability_weight", False)
getattr(cfg, "apply_reliability_weight", False)

# FORBIDDEN – hard-coded mode in repository
mode="continuous"   # must come from canonical config
```

## Invariants (enforced by tests)

1. Empty / missing config → `apply_reliability_weight is True`
2. Explicit `False` survives JSON → object → JSON round-trip
3. `risk_aversion` default `0.0` everywhere
4. Invalid `reliability_weight_mode` → config error (not coerced to bool)
5. SHADOW ≠ ACTIVE for production decision path
