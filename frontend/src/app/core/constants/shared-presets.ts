/**
 * Shared building blocks for the Optimizer / Auction preset catalogs.
 *
 * Recalibrated on Quotazioni Fantacalcio **2026/27** (primary) with continuity
 * checks against pooled 2023/24–2025/26 (sheet "Tutti", Qt.A):
 *
 * 2026/27 snapshot (n=497):
 *   mean≈6.54  median=5  p25=3  p75=8  p90≈14  p95≈17.2  max=35
 *   Qt.A=1 ≈17.3% (noise / pure reserves)
 *   usable ≥5 → 283 (≈57%)   solid ≥8 → 152 (≈31%)
 *   premium ≥12 → 73         elite ≥18 → 25
 *   ultra ≥25 → 13           ≥30 → 4
 *   full-elite 25-man listino cost (3P/8D/8C/6A top) ≈562 vs budget 500
 *   → concentration caps remain load-bearing; full-elite squads are over budget
 *
 * Role medians 2026/27: P≈1  D≈5  C≈5  A≈8
 * Big-team sum Qt.A order: Inter ≫ Milan > Napoli ≈ Como > Juve > Roma ≈ Atalanta
 *   → Como stays in DEFAULT_BIG_TEAMS (4th by aggregate listino).
 *
 * Structural shift vs 2023–25: lower mean, tighter mid-tier, similar ultra count.
 * Presets must not assume the old p95≈20–21 market.
 *
 * Everything here is `readonly`/`as const` on purpose: presets must never
 * mutate a shared array in place.
 */
import { FormationConfig } from '../models/api.models';
import { AuctionRole } from '../models/auction.models';

/**
 * "Big teams" for the hard bigTeamsCap constraint.
 * Core six + Como (still top-4 by aggregate Qt.A in 2026/27: Inter 294,
 * Milan 239, Napoli 227, Como 224).
 */
export const DEFAULT_BIG_TEAMS: string[] = [
  'Inter',
  'Milan',
  'Juventus',
  'Napoli',
  'Roma',
  'Atalanta',
  'Como',
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

/**
 * Official Mantra Experience 2026/27 module labels (11).
 * Used for preferredMantraFormation selects and coverage badges.
 * See docs/mantra_formations_2026_27.md.
 */
export const MANTRA_MODULE_LABELS: readonly string[] = [
  '3-4-3',
  '3-4-1-2',
  '3-4-2-1',
  '3-5-2',
  '3-5-1-1',
  '4-3-3',
  '4-3-1-2',
  '4-4-2',
  '4-1-4-1',
  '4-4-1-1',
  '4-2-3-1',
] as const;

// ---------------------------------------------------------------------------
// Empirical Qt.A anchors (primary: 2026/27; continuity with 2023–25)
// Used by preset authors to keep minQtA / topTierCostThreshold coherent.
// ---------------------------------------------------------------------------

/**
 * Reliability tiers derived from empirical Qt.A (2026/27 calibrated):
 * - NOISE (1): pure reserves / listed backups — filter for almost all strategies
 * - FRINGE (2–3): low reliability, rotation-only / lottery tickets
 * - USABLE (4–5): rotation / solid mid-tier (median = 5)
 * - SOLID (6–8): good starters / value core  (p50–p75)
 * - PREMIUM (9–14): high-quality semi-stars (≈p75–p90)
 * - ELITE (15–24): true top-tier (~25 names ≥18 in 2026/27)
 * - ULTRA (25+): rare superstars (13 names; 4 ≥30)
 */
export const QT_A_TIERS = {
  noise: 1,
  fringe: 3,
  usable: 5,
  solid: 8,
  premium: 12,
  elite: 18,
  ultra: 25,
} as const;

/**
 * Suggested topTierCostThreshold by aggressiveness (2026/27).
 *
 * Anchored to p95≈17.2 and elite floor ≥18 (25 names). Caps above 26 are
 * rarely binding; ultra (≥25) is still present but scarce.
 */
export const TOP_TIER_COST = {
  /** Underdog / pure value — almost no true premiums. */
  strict: 18,
  /** Floor / safe / anti-injury. */
  moderate: 22,
  /** Balanced / tournament / championship. */
  open: 26,
  /** Ceiling / risk-on — no hard cap. */
  free: null as number | null,
} as const;

/**
 * Empirical budget-share prior from top-quota listino cost 2026/27
 * (3P·top + 8D·top + 8C·top + 6A·top ≈ 562):
 *   P≈0.09  D≈0.24  C≈0.35  A≈0.32
 *
 * Auction strategies may deliberately overweight A (scoring leverage) or
 * C (Mantra flexibility); this prior is the neutral reference.
 */
export const LISTINO_BUDGET_SHARE_PRIOR: Readonly<Record<AuctionRole, number>> = {
  P: 0.09,
  D: 0.24,
  C: 0.35,
  A: 0.32,
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
