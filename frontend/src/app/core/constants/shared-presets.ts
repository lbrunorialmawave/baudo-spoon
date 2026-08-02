/**
 * Shared building blocks for the Optimizer / Auction preset catalogs.
 *
 * Recalibrated against Quotazioni Fantacalcio 2025/26 (Qt.A distribution):
 * - Overall Qt.A: mean≈8.0, median≈7, p75≈11, max=33
 * - By role median: P=1, D=6, C=8, A=10
 * - Qt.A=1 is mostly pure reserves (esp. GK); usable starters cluster ≥5–8
 * - FVM correlates strongly with Qt.A (r≈0.76) → higher Qt.A = higher reliability
 *
 * Everything here is `readonly`/`as const` on purpose: presets must never
 * mutate a shared array in place.
 */
import { FormationConfig } from '../models/api.models';
import { AuctionRole } from '../models/auction.models';

/** The 7 "big teams" used by every stock preset for the big-teams cap. */
export const DEFAULT_BIG_TEAMS: string[] = [
  'Inter',
  'Milan',
  'Juventus',
  'Napoli',
  'Roma',
  'Como',
  'Atalanta',
] as const;

/** Classic Fantacalcio role quotas (3P / 8D / 8C / 6A = 25 total). */
export const DEFAULT_CLASSIC_ROLE_QUOTAS: Readonly<Record<AuctionRole, number>> = {
  P: 3,
  D: 8,
  C: 8,
  A: 6,
} as const;

/**
 * The 6 formations offered by every stock optimizer preset.
 * Catalog of modules to evaluate feasibility for — not a hard solver constraint
 * unless `preferredFormation` is set.
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
// Empirical Qt.A anchors (Fantacalcio 2025/26)
// Used by preset authors to keep minQtA / topTierCostThreshold coherent.
// ---------------------------------------------------------------------------

/**
 * Reliability tiers derived from empirical Qt.A:
 * - NOISE (1): pure reserves / listed backups — filter for almost all strategies
 * - FRINGE (2–4): low reliability, rotation-only
 * - USABLE (5–7): rotation / solid mid-tier
 * - SOLID (8–11): good starters / value core
 * - PREMIUM (12–17): high-quality semi-stars
 * - ELITE (18+): true top-tier (≈30 players league-wide)
 */
export const QT_A_TIERS = {
  noise: 1,
  fringe: 3,
  usable: 5,
  solid: 8,
  premium: 12,
  elite: 18,
} as const;

/**
 * Suggested topTierCostThreshold by aggressiveness.
 * Anchored to empirical p80–p90 Qt.A mixed across roles (~12–20)
 * and historical cost of premium names (Qt.A 20–28 band).
 */
export const TOP_TIER_COST = {
  strict: 22,   // underdog / value — few true premiums
  moderate: 26, // balanced / floor
  open: 30,     // aggressive / ceiling
  free: null as number | null, // no cap
} as const;

// ---------------------------------------------------------------------------
// Guardrail helpers
// ---------------------------------------------------------------------------

const EPSILON = 1e-6;

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
