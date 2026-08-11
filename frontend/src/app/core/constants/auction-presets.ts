/**
 * Auction strategy presets — single source of truth for the Auction setup UI.
 *
 * Presets are **strategy-only**: they carry market drift, inflation, alternatives,
 * valuation mode, replacement method, minStartProbability, hybridBlend, and a
 * semantic BidderPolicy. League logistics (numParticipants, budgetInitial,
 * referenceBudget, roleQuotas, ruleset) are owned exclusively by the setup form
 * and must never be overwritten by applyPreset().
 *
 * Recalibrated on Quotazioni Fantacalcio **2026/27** (primary; n=497 Qt.A):
 * - mean≈6.54  median=5  p75=8  p90≈14  p95≈17.2  max=35
 * - Qt.A=1 ≈17.3% noise | usable ≥5 → 283 | solid ≥8 → 152
 * - elite ≥18 → 25 names | ultra ≥25 → 13 | full-elite 25-man cost ≈562 vs 500
 * - Role medians: P≈1  D≈5  C≈5  A≈8
 * - Listino prior (top-quota): P0.09 / D0.24 / C0.35 / A0.32
 *   Strategy budgetShare may overweight A (scoring leverage) but must not
 *   invent a market that no longer exists (mid-tier compressed vs 2023–25)
 * - Inflation: fewer mid-premiums ⇒ slightly lower maxInflation on aggressive
 *   profiles; value/late profiles stay tight (pctl ≥0.76)
 * - Big-team reality: Inter dominant; Como still top-4 aggregate → teamStrength
 *   remains relevant for aggressive / early-stars profiles
 *
 * @see AuctionConfig / InitializeAuctionRequest in core/models/auction.models.ts
 * @see LISTINO_BUDGET_SHARE_PRIOR / TOP_TIER_COST / QT_A_TIERS in shared-presets.ts
 */
import { AuctionConfig } from '../models/auction.models';

/** Semantic UI / auto-bidder hints — never sent as-is to POST /auction/init. */
export interface AuctionPresetPolicy {
  aggressiveness?: number;
  inflationTolerance?: number;
  maxOverpayRatio?: number;
  minResidualCreditsPerSlot?: number;
  allInProbability?: number;
  budgetElasticity?: number;
  varWeight?: number;
  teamStrengthWeight?: number;
  preferAlternatives?: boolean;
  preferLowCostAlternative?: boolean;
  rebidTriggerPctAboveExpected?: number;
  budgetShareByRole?: Partial<Record<'P' | 'D' | 'C' | 'A', number>>;
  phaseBias?: 'early' | 'late' | string;
  preferYoungPlayers?: boolean;
  maxAgePreference?: number;
  preferHighStartProbability?: boolean;
  minStartProbability?: number;
  preferHighVariance?: boolean;
  preferMultiRole?: boolean;
  minNumRoles?: number;
  budgetShareByBlock?: Record<string, number>;
  maxTopTierCount?: number;
  targetTopTierCount?: number;
  avoidTopTierEarly?: boolean;
  adaptive?: boolean;
  adaptOn?: string[];
}

/**
 * Strategy slice of AuctionConfig that presets are allowed to set.
 * Logistics (numParticipants, budgetInitial, referenceBudget, roleQuotas,
 * ruleset) are deliberately excluded — those belong to the setup form only.
 */
export type AuctionStrategyConfig = Pick<
  AuctionConfig,
  | 'marketDriftConfig'
  | 'alternativesConfig'
  | 'useInflationBaseline'
  | 'inflationConfig'
  | 'valuationMode'
  | 'replacementMethod'
  | 'minStartProbability'
  | 'hybridBlend'
>;

export interface AuctionPreset {
  readonly id: string;
  readonly name: string;
  readonly labelIt: string;
  readonly description: string;
  /**
   * Which ruleset(s) this preset is valid for. Used by the UI to filter/disable
   * incompatible presets instead of silently merging overlapping role keys.
   */
  readonly rulesetTarget: 'CLASSIC' | 'MANTRA' | 'BOTH';
  /** Strategy-only config applied to the setup form (never logistics). */
  readonly config: AuctionStrategyConfig;
  readonly policy: AuctionPresetPolicy;
}

/**
 * Identity function used purely for contextual type-checking of strategy config.
 * Prefer this over a cast in every new preset.
 */
function defineAuctionConfig(config: AuctionStrategyConfig): AuctionStrategyConfig {
  return config;
}

export const AUCTION_PRESET_NONE = '' as const;

export const AUCTION_PRESETS: readonly AuctionPreset[] = [
  {
    id: 'conservative',
    name: 'Conservative',
    labelIt: 'Conservativo',
    description:
      'Evita overpay, protegge il residuo, predilige alternative low-cost e inflazione bassa. Filtra il rumore Qt.A=1 (~17% del listino 2026/27). Ideale se sei ultimo di budget o temi di restare scoperto a fine asta.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.16,
        spilloverAdjacentTier: 0.1,
        spilloverCrossRole: 0.0,
        minIndex: 0.68,
        maxIndex: 1.3,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.28,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.8,
        maxInflationMultiplier: 1.28,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.03,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: 0.58,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.2,
      inflationTolerance: 0.22,
      maxOverpayRatio: 1.06,
      minResidualCreditsPerSlot: 3.5,
      allInProbability: 0.03,
      budgetElasticity: 0.15,
      varWeight: 0.2,
      teamStrengthWeight: 0.06,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.03,
      budgetShareByRole: {
        P: 0.09,
        D: 0.25,
        C: 0.32,
        A: 0.34,
      },
    },
  },
  {
    id: 'balanced',
    name: 'Balanced',
    labelIt: 'Bilanciato',
    description:
      'Profilo neutro allineato al prior listino 2026/27 (P0.09/D0.24/C0.35/A0.32 + leggero overweight A). EWMA standard, inflazione moderata. Punto di partenza consigliato per leghe a 8.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.28,
        spilloverAdjacentTier: 0.22,
        spilloverCrossRole: 0.0,
        minIndex: 0.55,
        maxIndex: 1.65,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.36,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.72,
        maxInflationMultiplier: 1.5,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.1,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.5,
      inflationTolerance: 0.48,
      maxOverpayRatio: 1.16,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.12,
      budgetElasticity: 0.4,
      varWeight: 0.35,
      teamStrengthWeight: 0.16,
      preferAlternatives: true,
      rebidTriggerPctAboveExpected: 0.12,
      budgetShareByRole: {
        P: 0.08,
        D: 0.23,
        C: 0.33,
        A: 0.36,
      },
    },
  },
  {
    id: 'aggressive',
    name: 'Aggressive',
    labelIt: 'Aggressivo',
    description:
      'Insegue i top (elite Qt.A ≥18: solo 25 nomi in 2026/27), tollera overpay e inflazione alta, spillover più reattivo. Rischia residuo basso ma punta al ceiling.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.4,
        spilloverAdjacentTier: 0.32,
        spilloverCrossRole: 0.05,
        minIndex: 0.5,
        maxIndex: 1.95,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.48,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.58,
        maxInflationMultiplier: 1.7,
        baseInflationRate: 0.065,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.2,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.82,
      inflationTolerance: 0.75,
      maxOverpayRatio: 1.4,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.3,
      budgetElasticity: 0.68,
      varWeight: 0.55,
      teamStrengthWeight: 0.3,
      preferAlternatives: false,
      rebidTriggerPctAboveExpected: 0.22,
      budgetShareByRole: {
        P: 0.06,
        D: 0.18,
        C: 0.3,
        A: 0.46,
      },
    },
  },
  {
    id: 'top_player_hunter',
    name: 'Top Player Hunter',
    labelIt: 'Cacciatore di top',
    description:
      'Concentra budget sui TOP tier (Qt.A ≥18 / ultra ≥25; 25+13 nomi). Spillover alto sui top, all-in più probabile. Accetta buchi di rosa per 2–4 nomi chiave.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.38,
        spilloverAdjacentTier: 0.38,
        spilloverCrossRole: 0.08,
        minIndex: 0.5,
        maxIndex: 2.0,
        tierThresholds: [0.3, 0.7],
      },
      alternativesConfig: {
        lowCostPercentile: 0.45,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.55,
        maxInflationMultiplier: 1.85,
        baseInflationRate: 0.075,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.26,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.9,
      inflationTolerance: 0.85,
      maxOverpayRatio: 1.52,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.4,
      budgetElasticity: 0.78,
      varWeight: 0.7,
      teamStrengthWeight: 0.36,
      preferAlternatives: false,
      rebidTriggerPctAboveExpected: 0.26,
      targetTopTierCount: 4,
      budgetShareByRole: {
        P: 0.05,
        D: 0.15,
        C: 0.28,
        A: 0.52,
      },
    },
  },
  {
    id: 'value_hunter',
    name: 'Value Hunter',
    labelIt: 'Cacciatore di valore',
    description:
      'Massimizza ESV/VR nella fascia usable–solid (Qt.A 5–12: cuore del listino 2026/27, mediana=5 p90≈14). Overpay quasi nullo; VAR e alternative low-cost pesano molto.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.2,
        spilloverAdjacentTier: 0.16,
        spilloverCrossRole: 0.0,
        minIndex: 0.6,
        maxIndex: 1.4,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.26,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.82,
        maxInflationMultiplier: 1.3,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.03,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.3,
      inflationTolerance: 0.2,
      maxOverpayRatio: 1.05,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.05,
      budgetElasticity: 0.2,
      varWeight: 0.8,
      teamStrengthWeight: 0.06,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.04,
      budgetShareByRole: {
        P: 0.09,
        D: 0.24,
        C: 0.33,
        A: 0.34,
      },
    },
  },
  {
    id: 'inflation_fighter',
    name: 'Inflation Fighter',
    labelIt: 'Anti-inflazione',
    description:
      'Assume mercato caldo: inflazione alta nel baseline, EWMA reattivo, non insegue i picchi TOP (solo 25 elite). Preferisce MID solidi (Qt.A 6–12) e alternative.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.46,
        spilloverAdjacentTier: 0.28,
        spilloverCrossRole: 0.06,
        minIndex: 0.5,
        maxIndex: 2.0,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.3,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.52,
        maxInflationMultiplier: 1.85,
        baseInflationRate: 0.085,
        baselineParticipants: 6,
        teamStrengthMultiplier: 0.22,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.36,
      inflationTolerance: 0.16,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.07,
      budgetElasticity: 0.26,
      varWeight: 0.42,
      teamStrengthWeight: 0.4,
      preferAlternatives: true,
      avoidTopTierEarly: true,
      rebidTriggerPctAboveExpected: 0.06,
      budgetShareByRole: {
        P: 0.09,
        D: 0.25,
        C: 0.33,
        A: 0.33,
      },
    },
  },
  {
    id: 'early_stars',
    name: 'Early Stars',
    labelIt: "Stelle all'inizio",
    description:
      'Spende forte nella prima fase sui nomi chiave (elite/ultra 2026/27), poi si adatta. Overpay consentito solo sui TOP; dopo fase 1 diventa più selettivo.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.34,
        spilloverAdjacentTier: 0.28,
        spilloverCrossRole: 0.05,
        minIndex: 0.5,
        maxIndex: 1.85,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.4,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.62,
        maxInflationMultiplier: 1.7,
        baseInflationRate: 0.055,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.18,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.72,
      inflationTolerance: 0.6,
      maxOverpayRatio: 1.35,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.26,
      budgetElasticity: 0.55,
      varWeight: 0.5,
      teamStrengthWeight: 0.26,
      preferAlternatives: false,
      phaseBias: 'early',
      rebidTriggerPctAboveExpected: 0.18,
      budgetShareByRole: {
        P: 0.06,
        D: 0.18,
        C: 0.3,
        A: 0.46,
      },
    },
  },
  {
    id: 'late_sniper',
    name: 'Late Sniper',
    labelIt: 'Cecchino di fine asta',
    description:
      'Conserva budget e agisce a fine asta su residual value nella fascia Qt.A 5–11 (usable–solid 2026/27). Alpha moderato, overpay basso, alternative low-cost prioritizzate.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.22,
        spilloverAdjacentTier: 0.18,
        spilloverCrossRole: 0.0,
        minIndex: 0.6,
        maxIndex: 1.5,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.24,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.78,
        maxInflationMultiplier: 1.35,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.05,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.26,
      inflationTolerance: 0.3,
      maxOverpayRatio: 1.06,
      minResidualCreditsPerSlot: 3.5,
      allInProbability: 0.1,
      budgetElasticity: 0.3,
      varWeight: 0.65,
      teamStrengthWeight: 0.1,
      preferAlternatives: true,
      phaseBias: 'late',
      rebidTriggerPctAboveExpected: 0.05,
      budgetShareByRole: {
        P: 0.09,
        D: 0.24,
        C: 0.33,
        A: 0.34,
      },
    },
  },
  {
    id: 'youth_first',
    name: 'Youth First',
    labelIt: 'Giovani first',
    description:
      'Privilegia season_value e potenziale (giovani in rampa). Drift moderato; team strength disattivato; tollera minutaggio incerto se ESV alto. Ideale con listino compresso dove i value young emergono.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.26,
        spilloverAdjacentTier: 0.16,
        spilloverCrossRole: 0.0,
        minIndex: 0.52,
        maxIndex: 1.6,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.34,
      },
      useInflationBaseline: false,
      inflationConfig: {
        inflationPercentileThreshold: 0.72,
        maxInflationMultiplier: 1.4,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.0,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.45,
      inflationTolerance: 0.35,
      maxOverpayRatio: 1.14,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.15,
      budgetElasticity: 0.42,
      varWeight: 0.58,
      teamStrengthWeight: 0.03,
      preferAlternatives: true,
      preferYoungPlayers: true,
      maxAgePreference: 23,
      rebidTriggerPctAboveExpected: 0.12,
      budgetShareByRole: {
        P: 0.07,
        D: 0.22,
        C: 0.34,
        A: 0.37,
      },
    },
  },
  {
    id: 'safe_picks',
    name: 'Safe Picks',
    labelIt: 'Scelte sicure',
    description:
      'Titolari consolidati (alta Pr, Qt.A solidi ≥8: 152 nomi in 2026/27), bassa varianza, overpay minimo. Ideale per chi vuole certezze e floor elevato.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.18,
        spilloverAdjacentTier: 0.12,
        spilloverCrossRole: 0.0,
        minIndex: 0.65,
        maxIndex: 1.35,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.34,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.76,
        maxInflationMultiplier: 1.35,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.08,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: 0.72,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.22,
      inflationTolerance: 0.25,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.04,
      budgetElasticity: 0.2,
      varWeight: 0.2,
      teamStrengthWeight: 0.16,
      preferAlternatives: true,
      preferHighStartProbability: true,
      minStartProbability: 0.72,
      rebidTriggerPctAboveExpected: 0.05,
      budgetShareByRole: {
        P: 0.09,
        D: 0.26,
        C: 0.31,
        A: 0.34,
      },
    },
  },
  {
    id: 'risk_lover',
    name: 'Risk Lover',
    labelIt: 'Amante del rischio',
    description:
      'Scommesse, high ceiling, outlier VAR. Season value, spillover cross-role, all-in frequente. Accetta Qt.A bassi se il potenziale è alto (pool noise 17% da filtrare solo a livello policy).',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.46,
        spilloverAdjacentTier: 0.34,
        spilloverCrossRole: 0.12,
        minIndex: 0.5,
        maxIndex: 2.0,
        tierThresholds: [0.28, 0.68],
      },
      alternativesConfig: {
        lowCostPercentile: 0.55,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.52,
        maxInflationMultiplier: 1.85,
        baseInflationRate: 0.075,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.12,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.92,
      inflationTolerance: 0.75,
      maxOverpayRatio: 1.48,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.48,
      budgetElasticity: 0.85,
      varWeight: 0.85,
      teamStrengthWeight: 0.1,
      preferAlternatives: false,
      preferHighVariance: true,
      rebidTriggerPctAboveExpected: 0.28,
      budgetShareByRole: {
        P: 0.05,
        D: 0.16,
        C: 0.3,
        A: 0.49,
      },
    },
  },
  {
    id: 'mantra_optimized',
    name: 'Mantra Optimized',
    labelIt: 'Ottimizzato Mantra',
    description:
      'Pensato per leghe MANTRA 2026/27: valorizza polivalenza (Num_Ruoli), flessibilità di modulo (11 moduli ufficiali), alternative multi-ruolo. Budget share allineato ai 6 blocchi algoritmo v3.1.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.28,
        spilloverAdjacentTier: 0.24,
        spilloverCrossRole: 0.1,
        minIndex: 0.52,
        maxIndex: 1.7,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.36,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.7,
        maxInflationMultiplier: 1.5,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.12,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.5,
      inflationTolerance: 0.48,
      maxOverpayRatio: 1.18,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.14,
      budgetElasticity: 0.45,
      varWeight: 0.45,
      teamStrengthWeight: 0.16,
      preferAlternatives: true,
      preferMultiRole: true,
      minNumRoles: 2,
      rebidTriggerPctAboveExpected: 0.12,
      budgetShareByBlock: {
        Por: 0.07,
        Difesa_Pura: 0.14,
        Ibridi_Difensivi: 0.12,
        Centro_Nevralgico: 0.18,
        Linea_Fantasia: 0.18,
        Attacco: 0.31,
      },
      budgetShareByRole: {
        P: 0.07,
        D: 0.28,
        C: 0.33,
        A: 0.32,
      },
    },
  },
  {
    id: 'depth_builder',
    name: 'Depth Builder',
    labelIt: 'Costruttore di profondità',
    description:
      'Priorità a copertura ruoli e panchina utile (Qt.A usable 5–10: ~230 nomi) piuttosto che a un singolo superstar. Budget più uniforme, residuale alto. maxTopTierCount=2.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.22,
        spilloverAdjacentTier: 0.16,
        spilloverCrossRole: 0.0,
        minIndex: 0.6,
        maxIndex: 1.45,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.28,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.78,
        maxInflationMultiplier: 1.32,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.05,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.3,
      inflationTolerance: 0.3,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.03,
      budgetElasticity: 0.26,
      varWeight: 0.3,
      teamStrengthWeight: 0.1,
      preferAlternatives: true,
      maxTopTierCount: 2,
      rebidTriggerPctAboveExpected: 0.06,
      budgetShareByRole: {
        P: 0.09,
        D: 0.25,
        C: 0.33,
        A: 0.33,
      },
    },
  },
  {
    id: 'budget_saver',
    name: 'Budget Saver',
    labelIt: 'Risparmiatore',
    description:
      'Massimizza residuo e flessibilità di fine asta. Quasi zero overpay; credit reserve stretta. Filtra il rumore e punta su low-cost solidi (Qt.A usable senza elite).',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.14,
        spilloverAdjacentTier: 0.08,
        spilloverCrossRole: 0.0,
        minIndex: 0.7,
        maxIndex: 1.25,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.2,
      },
      useInflationBaseline: false,
      inflationConfig: {
        inflationPercentileThreshold: 0.86,
        maxInflationMultiplier: 1.2,
        baseInflationRate: 0.02,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.0,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.1,
      inflationTolerance: 0.1,
      maxOverpayRatio: 1.03,
      minResidualCreditsPerSlot: 4.5,
      allInProbability: 0.015,
      budgetElasticity: 0.06,
      varWeight: 0.4,
      teamStrengthWeight: 0.03,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.025,
      budgetShareByRole: {
        P: 0.09,
        D: 0.26,
        C: 0.33,
        A: 0.32,
      },
    },
  },
  {
    id: 'adaptive_ai',
    name: 'Adaptive AI',
    labelIt: 'AI adattiva',
    description:
      'Profilo meta 2026/27: EWMA reattivo, VAR e team strength bilanciati, valuation season, inflazione attiva. Si aggiorna dinamicamente su price_index e residuo.',
    rulesetTarget: 'CLASSIC',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.34,
        spilloverAdjacentTier: 0.26,
        spilloverCrossRole: 0.06,
        minIndex: 0.5,
        maxIndex: 1.8,
        tierThresholds: [0.38, 0.78],
      },
      alternativesConfig: {
        lowCostPercentile: 0.36,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.64,
        maxInflationMultiplier: 1.65,
        baseInflationRate: 0.055,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.18,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.56,
      inflationTolerance: 0.5,
      maxOverpayRatio: 1.22,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.18,
      budgetElasticity: 0.55,
      varWeight: 0.6,
      teamStrengthWeight: 0.26,
      preferAlternatives: true,
      adaptive: true,
      adaptOn: ['price_index', 'budget_residual', 'role_fill_rate'],
      rebidTriggerPctAboveExpected: 0.15,
      budgetShareByRole: {
        P: 0.07,
        D: 0.22,
        C: 0.33,
        A: 0.38,
      },
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // MANTRA-only strategy presets (2026/27)
  // ─────────────────────────────────────────────────────────────────────────

  {
    id: 'mantra_flex_master',
    name: 'Mantra Flex Master',
    labelIt: 'Maestro di flessibilità',
    description:
      'Massimizza polivalenza e copertura dei 11 moduli ufficiali. Preferisce multi-ruolo (minNumRoles≥2), budget bilanciato sui 6 blocchi algoritmo v3.1, residuale alto per riempire slot scarsi (T/W/E/Ds/B) a fine asta. Ideale per leghe Mantra a 8.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.26,
        spilloverAdjacentTier: 0.22,
        spilloverCrossRole: 0.12,
        minIndex: 0.55,
        maxIndex: 1.65,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.34,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.72,
        maxInflationMultiplier: 1.45,
        baseInflationRate: 0.045,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.1,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.48,
      inflationTolerance: 0.45,
      maxOverpayRatio: 1.15,
      minResidualCreditsPerSlot: 1.8,
      allInProbability: 0.1,
      budgetElasticity: 0.42,
      varWeight: 0.4,
      teamStrengthWeight: 0.14,
      preferAlternatives: true,
      preferMultiRole: true,
      minNumRoles: 2,
      rebidTriggerPctAboveExpected: 0.1,
      budgetShareByBlock: {
        Por: 0.07,
        Difesa_Pura: 0.15,
        Ibridi_Difensivi: 0.13,
        Centro_Nevralgico: 0.17,
        Linea_Fantasia: 0.17,
        Attacco: 0.31,
      },
      budgetShareByRole: {
        P: 0.07,
        D: 0.28,
        C: 0.34,
        A: 0.31,
      },
    },
  },
  {
    id: 'mantra_stars_first',
    name: 'Mantra Stars First',
    labelIt: 'Stelle Mantra first',
    description:
      'Insegue i top multi-ruolo di Attacco e Linea Fantasia (T/W/A/Pc). Tolleranza overpay e inflazione alta, all-in frequente. Accetta rosa meno profonda in difesa per 3–5 nomi ceiling. Spillover cross-role attivo.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.38,
        spilloverAdjacentTier: 0.3,
        spilloverCrossRole: 0.14,
        minIndex: 0.5,
        maxIndex: 1.9,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.45,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.6,
        maxInflationMultiplier: 1.68,
        baseInflationRate: 0.06,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.22,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.8,
      inflationTolerance: 0.72,
      maxOverpayRatio: 1.38,
      minResidualCreditsPerSlot: 1.0,
      allInProbability: 0.28,
      budgetElasticity: 0.65,
      varWeight: 0.55,
      teamStrengthWeight: 0.28,
      preferAlternatives: false,
      preferMultiRole: true,
      minNumRoles: 2,
      maxTopTierCount: 5,
      rebidTriggerPctAboveExpected: 0.2,
      budgetShareByBlock: {
        Por: 0.05,
        Difesa_Pura: 0.1,
        Ibridi_Difensivi: 0.1,
        Centro_Nevralgico: 0.15,
        Linea_Fantasia: 0.22,
        Attacco: 0.38,
      },
      budgetShareByRole: {
        P: 0.05,
        D: 0.18,
        C: 0.28,
        A: 0.49,
      },
    },
  },
  {
    id: 'mantra_defensive_wall',
    name: 'Mantra Defensive Wall',
    labelIt: 'Muro difensivo Mantra',
    description:
      'Priorità assoluta a Por + Difesa Pura + Ibridi Difensivi. Preferisce alta probabilità di titolarità, bassa varianza, multi-ruolo difensivi (Dc/B, E/M). Budget difensivo sovrappesato; attacco low-cost. Ideale per moduli a 3 o 5 difensori.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.24,
        spilloverAdjacentTier: 0.18,
        spilloverCrossRole: 0.08,
        minIndex: 0.58,
        maxIndex: 1.55,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.3,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.76,
        maxInflationMultiplier: 1.38,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.08,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: 0.55,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.42,
      inflationTolerance: 0.38,
      maxOverpayRatio: 1.12,
      minResidualCreditsPerSlot: 2.0,
      allInProbability: 0.08,
      budgetElasticity: 0.35,
      varWeight: 0.25,
      teamStrengthWeight: 0.12,
      preferAlternatives: true,
      preferMultiRole: true,
      minNumRoles: 2,
      preferHighStartProbability: true,
      minStartProbability: 0.55,
      rebidTriggerPctAboveExpected: 0.08,
      budgetShareByBlock: {
        Por: 0.1,
        Difesa_Pura: 0.22,
        Ibridi_Difensivi: 0.16,
        Centro_Nevralgico: 0.16,
        Linea_Fantasia: 0.12,
        Attacco: 0.24,
      },
      budgetShareByRole: {
        P: 0.1,
        D: 0.36,
        C: 0.28,
        A: 0.26,
      },
    },
  },
  {
    id: 'mantra_value_poly',
    name: 'Mantra Value Poly',
    labelIt: 'Valore polivalente',
    description:
      'Cacciatore di valore adattato al Mantra: low-cost multi-ruolo, inflazione stretta, residuale alto. Punta su giocatori con ≥2 ruoli e Qt.A solidi senza elite. Completa la rosa senza buchi e massimizza flessibilità di fine asta.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.2,
        spilloverAdjacentTier: 0.14,
        spilloverCrossRole: 0.08,
        minIndex: 0.62,
        maxIndex: 1.4,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.28,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.8,
        maxInflationMultiplier: 1.28,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.05,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.32,
      inflationTolerance: 0.28,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 2.4,
      allInProbability: 0.04,
      budgetElasticity: 0.28,
      varWeight: 0.35,
      teamStrengthWeight: 0.08,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      preferMultiRole: true,
      minNumRoles: 2,
      maxTopTierCount: 2,
      rebidTriggerPctAboveExpected: 0.05,
      budgetShareByBlock: {
        Por: 0.08,
        Difesa_Pura: 0.16,
        Ibridi_Difensivi: 0.14,
        Centro_Nevralgico: 0.18,
        Linea_Fantasia: 0.16,
        Attacco: 0.28,
      },
      budgetShareByRole: {
        P: 0.08,
        D: 0.28,
        C: 0.34,
        A: 0.3,
      },
    },
  },
  {
    id: 'mantra_formation_builder',
    name: 'Mantra Formation Builder',
    labelIt: 'Costruttore di moduli',
    description:
      'Orientato alla schierabilità di 2–3 moduli core (3-5-2 / 4-3-3 / 4-2-3-1). Sovrappesa leggermente gli slot scarsi (T, W, E, Ds, B) e i multi-ruolo che sbloccano OR-group. Residual e alternative alti per chiudere i buchi di coverage.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.27,
        spilloverAdjacentTier: 0.22,
        spilloverCrossRole: 0.12,
        minIndex: 0.55,
        maxIndex: 1.68,
        tierThresholds: [0.38, 0.78],
      },
      alternativesConfig: {
        lowCostPercentile: 0.35,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.7,
        maxInflationMultiplier: 1.48,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.12,
      },
      valuationMode: 'SEASON_VALUE',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.52,
      inflationTolerance: 0.5,
      maxOverpayRatio: 1.18,
      minResidualCreditsPerSlot: 1.6,
      allInProbability: 0.12,
      budgetElasticity: 0.48,
      varWeight: 0.42,
      teamStrengthWeight: 0.16,
      preferAlternatives: true,
      preferMultiRole: true,
      minNumRoles: 2,
      rebidTriggerPctAboveExpected: 0.11,
      budgetShareByBlock: {
        Por: 0.07,
        Difesa_Pura: 0.16,
        Ibridi_Difensivi: 0.14,
        Centro_Nevralgico: 0.16,
        Linea_Fantasia: 0.2,
        Attacco: 0.27,
      },
      budgetShareByRole: {
        P: 0.07,
        D: 0.28,
        C: 0.32,
        A: 0.33,
      },
    },
  },
  {
    id: 'mantra_late_flex',
    name: 'Mantra Late Flex',
    labelIt: 'Flessibilità di fine asta',
    description:
      'Cecchino Mantra: aggressività bassa all’inizio, residuale molto alto, alternative multi-ruolo. Aspetta l’inflazione sui top e chiude gli slot scarsi (T/W/E/B/Ds) a prezzi di saldo. Completamento rosa prioritario.',
    rulesetTarget: 'MANTRA',
    config: defineAuctionConfig({
      marketDriftConfig: {
        alpha: 0.18,
        spilloverAdjacentTier: 0.12,
        spilloverCrossRole: 0.06,
        minIndex: 0.65,
        maxIndex: 1.35,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.25,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.82,
        maxInflationMultiplier: 1.25,
        baseInflationRate: 0.025,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.04,
      },
      valuationMode: 'PER_MATCH_RATING',
      minStartProbability: null,
      replacementMethod: 'percentile',
    }),
    policy: {
      aggressiveness: 0.28,
      inflationTolerance: 0.25,
      maxOverpayRatio: 1.06,
      minResidualCreditsPerSlot: 2.8,
      allInProbability: 0.03,
      budgetElasticity: 0.22,
      varWeight: 0.3,
      teamStrengthWeight: 0.06,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      preferMultiRole: true,
      minNumRoles: 2,
      phaseBias: 'late',
      avoidTopTierEarly: true,
      maxTopTierCount: 1,
      rebidTriggerPctAboveExpected: 0.04,
      budgetShareByBlock: {
        Por: 0.08,
        Difesa_Pura: 0.16,
        Ibridi_Difensivi: 0.14,
        Centro_Nevralgico: 0.18,
        Linea_Fantasia: 0.16,
        Attacco: 0.28,
      },
      budgetShareByRole: {
        P: 0.08,
        D: 0.28,
        C: 0.34,
        A: 0.3,
      },
    },
  },
] as const;

export const AUCTION_PRESETS_BY_ID: ReadonlyMap<string, AuctionPreset> =
  new Map(AUCTION_PRESETS.map(p => [p.id, p]));

export function findAuctionPreset(id: string): AuctionPreset | undefined {
  return AUCTION_PRESETS_BY_ID.get(id);
}
