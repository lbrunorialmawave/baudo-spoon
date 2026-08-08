/**
 * Shared building blocks for the Optimizer / Auction preset catalogs.
 *
 * Recalibrated on Quotazioni Fantacalcio pooled across 2023/24–2025/26
 * (sheet "Tutti", Qt.A):
 *
 * Overall (pooled ~1.6k rows):
 *   mean≈8.1  median=7  p75≈11–12  p90≈16  p95≈20–21  max 41→40→33
 *
 * By role (pooled medians):
 *   P≈1–3   D≈5–6   C≈8   A≈10–12
 *
 * Structure:
 *   - Qt.A=1 ≈ 13–16% of the list (mostly GK reserves / pure noise)
 *   - usable starters cluster ≥5 (≈63–67% of pool)
 *   - solid core ≥8 (≈44–47%)
 *   - elite ≥18: 49 → 49 → 39 names (market flattened in 2025/26)
 *   - ultra-elite ≥26: 17 → 10 → 7  |  ≥30: 8 → 7 → 3
 *   - corr(Qt.A, FVM) ≈ 0.76–0.81 → higher Qt.A ≈ higher reliability
 *   - top-25 quota-cost (3P/8D/8C/6A) ≈ 553–592 vs budget 500 → full-elite
 *     squads are structurally over budget; presets must not assume it
 *
 * Everything here is `readonly`/`as const` on purpose: presets must never
 * mutate a shared array in place.
 */
import { FormationConfig } from '../models/api.models';
import { AuctionRole } from '../models/auction.models';

/**
 * "Big teams" for the hard bigTeamsCap constraint.
 * Core six are stable top-sum Qt.A across 2023–25; Como is included because
 * in 2025/26 it jumped into the top-4 by aggregate listino (Paz / Douvikas /
 * Baturina) and is the current-season market reality the UI optimises for.
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
// Empirical Qt.A anchors (pooled 2023/24 – 2025/26)
// Used by preset authors to keep minQtA / topTierCostThreshold coherent.
// ---------------------------------------------------------------------------

/**
 * Reliability tiers derived from empirical Qt.A:
 * - NOISE (1): pure reserves / listed backups — filter for almost all strategies
 * - FRINGE (2–4): low reliability, rotation-only
 * - USABLE (5–7): rotation / solid mid-tier
 * - SOLID (8–11): good starters / value core  (≈p50–p75)
 * - PREMIUM (12–17): high-quality semi-stars (≈p75–p90)
 * - ELITE (18–24): true top-tier (~40–50 names historically; 39 in 2025/26)
 * - ULTRA (25+): rare superstars (≤17 names/season; ≤7 in 2025/26)
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
 * Suggested topTierCostThreshold by aggressiveness.
 *
 * Anchored to pooled p95≈20–21 and the observed compression of the ultra band
 * in 2025/26 (max 33, only 3 names ≥30). Caps above 28 are almost never
 * binding on the current listino and only inflate "premium" counts.
 */
export const TOP_TIER_COST = {
  /** Underdog / pure value — almost no true premiums. */
  strict: 20,
  /** Floor / safe / anti-injury. */
  moderate: 24,
  /** Balanced / tournament / championship. */
  open: 28,
  /** Ceiling / risk-on — no hard cap. */
  free: null as number | null,
} as const;

/**
 * Empirical budget-share prior from quota-weighted mean Qt.A
 * (3P·5.3 + 8D·6.3 + 8C·8.9 + 6A·11.7 ≈ 208 listino points):
 *   P≈0.08  D≈0.24  C≈0.34  A≈0.34
 *
 * Auction strategies may deliberately overweight A (scoring leverage) or
 * C (Mantra flexibility); this prior is the neutral reference.
 */
export const LISTINO_BUDGET_SHARE_PRIOR: Readonly<Record<AuctionRole, number>> = {
  P: 0.08,
  D: 0.24,
  C: 0.34,
  A: 0.34,
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
