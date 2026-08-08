# Mantra Formations 2026/27 — Mapping Table & ADR

**Status:** Frozen for implementation (Phase 0 sign-off)  
**Catalog version:** `MANTRA_FORMATIONS_V2026_27`  
**Source of truth (code):** `ml/optimizer/formations.py`  
**Related plan:** Mantra Formation Catalog & Feasibility in Optimizer + Auction

---

## ADR summary

**Decision:** Mantra formations are **coverage constraints / checks**, not replacements for squad role quotas.

| Concern | Choice |
|---------|--------|
| Roster budget | Still driven by `MANTRA_DEFAULT_QUOTAS` / `mantra_role_quotas` (25 players) |
| Starting-XI shape | `MantraFormation` slot OR-groups; squad must be able to assign players without double-counting |
| Default behaviour | Post-hoc evaluation only (`enforce_preferred_mantra_formation=False`) |
| Hard constraint | Opt-in via `preferred_mantra_formation` + `enforce_preferred_mantra_formation=True` |
| Por | Implicit: every module requires ≥1 Por in squad; Por is **not** listed in outfield slots |
| Catalog scope | Official 11 modules only (no user-defined modules in v1) |
| Shared logic | Pure `evaluate_coverage` / `evaluate_all_coverages` used by Optimizer and Auction |

**Consequences:** Quotas and modules coexist. A squad can fill quotas but fail a module (reported as deficits). Enforcing a module never runs unless the operator opts in.

---

## Canonical role codes

Aligned with `ml/mantra/roles.py` / DB migration 006:

`Por, Dc, B, Dd, Ds, E, M, C, T, W, A, Pc`

---

## OR-group legend (frozen semantics)

| Image / UI label | Role set | Notes |
|------------------|----------|--------|
| Dc | `{Dc}` | Pure central |
| DC/B | `{Dc, B}` | Third centre-back in a back-three; B may fill |
| Dd | `{Dd}` | |
| Ds | `{Ds}` | |
| E | `{E}` | |
| E/W | `{E, W}` | |
| M | `{M}` | |
| M/C | `{M, C}` | |
| C | `{C}` | |
| C/T | `{C, T}` | |
| T | `{T}` | |
| T/A | `{T, A}` | |
| T/A/Pc | `{T, A, Pc}` | Used in 4-3-1-2 attack support |
| W | `{W}` | |
| W/A | `{W, A}` | |
| W/T | `{W, T}` | |
| A/Pc | `{A, Pc}` | |

**Assignment rule:** one player fills at most one slot. Matching is exact (bipartite / ILP), not greedy-only.

---

## Official catalog (11 modules)

Outfield slots always sum to **10**. Por is checked separately.

| Label | DEF | MID | TRQ / ATT |
|-------|-----|-----|-----------|
| **3-4-3** | Dc×2, DC/B×1 | E×2, M/C×1, C×1 | W/A×2, A/Pc×1 |
| **3-4-1-2** | Dc×2, DC/B×1 | E×2, M/C×1, C×1 | T×1; A/Pc×2 |
| **3-4-2-1** | Dc×2, DC/B×1 | M×1, M/C×1, E×1, E/W×1 | T×1, T/A×1; A/Pc×1 |
| **3-5-2** | Dc×2, DC/B×1 | M×1, M/C×1, C×1, E×1, E/W×1 | A/Pc×2 |
| **3-5-1-1** | Dc×2, DC/B×1 | M×2, C×1, E/W×2 | T/A×1; A/Pc×1 |
| **4-3-3** | Dd×1, Dc×2, Ds×1 | M/C×1, M×1, C×1 | W/A×2, A/Pc×1 |
| **4-3-1-2** | Dd×1, Dc×2, Ds×1 | M/C×1, M×1, C×1 | T×1; T/A/Pc×1, A/Pc×1 |
| **4-4-2** | Dd×1, Dc×2, Ds×1 | M/C×1, C×1, E×1, E/W×1 | A/Pc×2 |
| **4-1-4-1** | Dd×1, Dc×2, Ds×1 | M×1, C/T×1, T×1, E/W×1, W×1 | A/Pc×1 |
| **4-4-1-1** | Dd×1, Dc×2, Ds×1 | M×1, C×1, E/W×2 | T/A×1; A/Pc×1 |
| **4-2-3-1** | Dd×1, Dc×2, Ds×1 | M×1, M/C×1 | W/T×1, T×1, W/A×1; A/Pc×1 |

### Evidence / sources used for freeze

- Design plan appendix skeleton (internal)
- Public descriptions consistent with Fantacalcio Mantra module set (11 modules; 5 defensive + 5 offensive movement players rule in official regolamento)
- [Regolamento Sistema Mantra](https://www.fantacalcio.it/regolamenti/sistema-mantra) — roles including B, 5+5 movement split
- Community breakdowns (pazzidifanta, 90min) for dual-label interpretation; where they conflict, plan skeleton + “B in back-three” rule wins

### Ambiguities closed in this freeze

1. **DC/B** — always `{Dc, B}` for the flexible central in back-three modules (not Dc-only).
2. **Por** — never listed in outfield slots; coverage requires ≥1 Por when `require_por=True`.
3. **T/A/Pc** — kept for 4-3-1-2 as in the plan skeleton.
4. **No per-league overrides** in v1 — single official catalog.
5. **Auction** — coverage is informational only (does not block bids).

If Fantacalcio publishes a different official image mid-season, bump catalog version and migrate with a changelog note.

---

## Operator note: *quote di rosa* vs *schierabilità modulo*

- **Quote di rosa** = how many players of each Mantra role you must own (auction / optimizer budget constraint).
- **Schierabilità modulo** = whether some assignment of those players’ `eligible_roles` can fill every slot of a given module for a starting XI.

Both are required for a complete Mantra experience; neither replaces the other.

---

## Implementation checklist (post Phase 0)

- [x] `ml/optimizer/formations.py` catalog matches this table
- [x] Optimizer post-hoc + optional hard enforce
- [x] Auction summary residual coverage
- [x] Frontend badges / presets (Phase 5)
- [ ] Changelog entry for operators
