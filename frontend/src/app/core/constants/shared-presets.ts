/**
 * Shared building blocks for the Optimizer / Auction preset catalogs.
 *
 * Rationale (extracted after a review of `optimizer-presets.ts` and
 * `auction-presets.ts`): every preset repeated the same 6-formation list,
 * the same 4 "big teams" array and the same 4-role quota map verbatim.
 * With 16 optimizer presets and 15 auction presets that was ~450 lines of
 * pure duplication and — more importantly — a single source of truth
 * violation: changing the default formation catalog, the big-teams list,
 * or the classic role quotas required touching 30+ call sites, and it is
 * exactly the kind of change a reviewer will miss in a large diff.
 *
 * Everything here is `readonly`/`as const` on purpose: presets must never
 * mutate a shared array in place (a single `.push()` on a preset's
 * `formations` array would silently corrupt every other preset sharing
 * the same reference).
 */

import { FormationConfig } from '../models/api.models';
import { AuctionRole } from '../models/auction.models';

/** The 4 "big teams" used by every stock preset for the big-teams cap. */
export const DEFAULT_BIG_TEAMS:  string[] = [
  'Inter',
  'Milan',
  'Juventus',
  'Napoli',
] as const;

/** Classic Fantacalcio role quotas (3P / 8D / 8C / 6A = 25 total). */
export const DEFAULT_CLASSIC_ROLE_QUOTAS: Readonly<Record<AuctionRole, number>> = {
  P: 3,
  D: 8,
  C: 8,
  A: 6,
} as const;

/**
 * The 6 formations offered by every stock optimizer preset. This is the
 * *catalog of modules to evaluate feasibility for* (see
 * `OptimizationResult.formationFeasibility`); it is NOT a hard solver
 * constraint by itself — only `preferredFormation` is.
 */
export const DEFAULT_FORMATIONS: FormationConfig[] = [
  { label: '3-4-3', defenders: 3, midfielders: 4, forwards: 3 },
  { label: '3-5-2', defenders: 3, midfielders: 5, forwards: 2 },
  { label: '4-3-3', defenders: 4, midfielders: 3, forwards: 3 },
  { label: '4-4-2', defenders: 4, midfielders: 4, forwards: 2 },
  { label: '4-5-1', defenders: 4, midfielders: 5, forwards: 1 },
  { label: '5-3-2', defenders: 5, midfielders: 3, forwards: 2 },
] as const;

// ---------------------------------------------------------------------------
// Guardrail helpers — invariants the *backend* would otherwise reject at
// runtime (500 on a malformed request) or silently mis-price (a budget
// share that doesn't sum to 1). Exercised by
// `optimizer-presets.spec.ts` / `auction-presets.spec.ts` so a future
// preset author gets a fast, readable failure in CI instead of a solver
// 400 discovered by an end user mid-auction.
// ---------------------------------------------------------------------------

const EPSILON = 1e-6;

/**
 * Mirrors `StrategyProfile.__post_init__` in `ml/optimizer/models.py`:
 * every role weight map must define all 4 classic roles or the backend
 * raises `ValueError` and the whole strategy is rejected.
 */
export function assertRoleWeightComplete(
  roleWeight: Record<string, number>,
  context: string,
): void {
  for (const role of ['P', 'D', 'C', 'A']) {
    if (!(role in roleWeight)) {
      throw new Error(`[${context}] roleWeight is missing role '${role}'`);
    }
    if (roleWeight[role] < 0) {
      throw new Error(`[${context}] roleWeight['${role}'] must be >= 0`);
    }
  }
}

/**
 * Budget shares that don't sum to ~1.0 don't fail loudly anywhere — the
 * auto-bidder just silently under- or over-allocates. Enforced here as a
 * lint-time invariant rather than relying on manual review of 15 presets.
 */
export function assertBudgetShareSumsToOne(
  budgetShareByRole: Partial<Record<AuctionRole, number>> | undefined,
  context: string,
  epsilon = 1e-2,
): void {
  if (!budgetShareByRole) return;
  const sum = Object.values(budgetShareByRole).reduce((a, b) => a + (b ?? 0), 0);
  if (Math.abs(sum - 1.0) > epsilon) {
    throw new Error(
      `[${context}] budgetShareByRole sums to ${sum.toFixed(4)}, expected ~1.0`,
    );
  }
}

/** Mirrors the backend `mantra_role_quotas must sum to TOTAL_SQUAD_SIZE` check. */
export function assertMantraQuotasSumTo25(
  quotas: Record<string, number> | null | undefined,
  context: string,
): void {
  if (!quotas) return;
  const sum = Object.values(quotas).reduce((a, b) => a + b, 0);
  if (sum !== 25) {
    throw new Error(`[${context}] mantraRoleQuotas sums to ${sum}, expected 25`);
  }
}

/** Mirrors `MarketDriftConfig.__post_init__`: 0 <= low < top <= 1. */
export function assertTierThresholdsValid(
  thresholds: readonly [number, number],
  context: string,
): void {
  const [low, top] = thresholds;
  if (!(low >= 0 && low < top && top <= 1)) {
    throw new Error(
      `[${context}] tierThresholds must satisfy 0 <= low < top <= 1, got [${low}, ${top}]`,
    );
  }
}