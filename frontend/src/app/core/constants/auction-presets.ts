/**
 * Auction strategy presets — single source of truth for the Auction setup UI.
 *
 * Each preset.config mirrors AuctionConfig (camelCase) and is applied onto
 * the setup form fields. Operator-owned fields (seasonStart, participants)
 * are intentionally left untouched.
 *
 * Refactor notes:
 * - `roleQuotas` (3P/8D/8C/6A) was repeated verbatim in all 15 presets;
 *   it now comes from `preset-shared.constants.ts`.
 * - Every `config` object used to be built as a plain object literal
 *   suffixed with `as AuctionConfig`. A blind `as` cast does NOT type-check
 *   the literal against the target type — it silently *widens* whatever
 *   shape you wrote to `AuctionConfig`, so a typo'd field name or a
 *   `tierThresholds: [0.4, 0.8, 0.9]` (3 elements instead of the required
 *   tuple of 2) would compile without complaint and only blow up at
 *   runtime inside `MarketDriftConfig.__post_init__` on the backend.
 *   `defineAuctionConfig` below is the identity function typed as
 *   `(config: AuctionConfig) => AuctionConfig`; passing an object literal
 *   as its argument gives real contextual type-checking (excess-property
 *   checks included) with zero runtime cost.
 *
 * @see AuctionConfig / InitializeAuctionRequest in core/models/auction.models.ts
 * @see artifacts/profiles/auction_profiles.json
 */

import { AuctionConfig } from '../models/auction.models';
import { DEFAULT_CLASSIC_ROLE_QUOTAS } from './shared-presets';

/**
 * Identity function used purely for contextual type-checking (see refactor
 * note above). Prefer this over `obj as AuctionConfig` in every new preset.
 */
function defineAuctionConfig(config: AuctionConfig): AuctionConfig {
  return config;
}

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

export interface AuctionPreset {
  readonly id: string;
  readonly name: string;
  readonly labelIt: string;
  readonly description: string;
  /** Full AuctionConfig applied to the setup form. */
  readonly config: AuctionConfig;
  readonly policy: AuctionPresetPolicy;
}

export const AUCTION_PRESET_NONE = '' as const;

export const AUCTION_PRESETS: readonly AuctionPreset[] = [
  {
    id: "conservative",
    name: "Conservative",
    labelIt: "Conservativo",
    description: "Evita overpay, protegge il residuo, predilige alternative low-cost e inflazione bassa. Ideale se sei ultimo di budget o temi di restare scoperto a fine asta.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.2,
        spilloverAdjacentTier: 0.15,
        spilloverCrossRole: 0.0,
        minIndex: 0.6,
        maxIndex: 1.4,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.35,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.75,
        maxInflationMultiplier: 1.35,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.05,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.25,
      inflationTolerance: 0.3,
      maxOverpayRatio: 1.1,
      minResidualCreditsPerSlot: 2,
      allInProbability: 0.05,
      budgetElasticity: 0.2,
      varWeight: 0.2,
      teamStrengthWeight: 0.1,
      preferAlternatives: true,
      rebidTriggerPctAboveExpected: 0.05,
      budgetShareByRole: {
        P: 0.08,
        D: 0.22,
        C: 0.28,
        A: 0.42,
      },
    },
  },
  {
    id: "balanced",
    name: "Balanced",
    labelIt: "Bilanciato",
    description: "Profilo neutro: EWMA standard, inflazione moderata, alternative standard. Punto di partenza consigliato per la maggior parte delle leghe.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.3,
        spilloverAdjacentTier: 0.25,
        spilloverCrossRole: 0.0,
        minIndex: 0.5,
        maxIndex: 1.8,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.4,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.7,
        maxInflationMultiplier: 1.6,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.1,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.5,
      inflationTolerance: 0.5,
      maxOverpayRatio: 1.2,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.12,
      budgetElasticity: 0.4,
      varWeight: 0.35,
      teamStrengthWeight: 0.2,
      preferAlternatives: true,
      rebidTriggerPctAboveExpected: 0.12,
      budgetShareByRole: {
        P: 0.06,
        D: 0.2,
        C: 0.3,
        A: 0.44,
      },
    },
  },
  {
    id: "aggressive",
    name: "Aggressive",
    labelIt: "Aggressivo",
    description: "Insegue i top, tollera overpay e inflazione alta, spillover più reattivo. Rischia residuo basso ma punta a massimizzare il ceiling della rosa.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.45,
        spilloverAdjacentTier: 0.35,
        spilloverCrossRole: 0.05,
        minIndex: 0.5,
        maxIndex: 2.0,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.5,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.6,
        maxInflationMultiplier: 1.9,
        baseInflationRate: 0.07,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.2,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.85,
      inflationTolerance: 0.8,
      maxOverpayRatio: 1.45,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.35,
      budgetElasticity: 0.7,
      varWeight: 0.55,
      teamStrengthWeight: 0.35,
      preferAlternatives: false,
      rebidTriggerPctAboveExpected: 0.25,
      budgetShareByRole: {
        P: 0.05,
        D: 0.18,
        C: 0.27,
        A: 0.5,
      },
    },
  },
  {
    id: "top_player_hunter",
    name: "Top Player Hunter",
    labelIt: "Cacciatore di top",
    description: "Concentra budget sui TOP tier; spillover alto sui top, all-in più probabile sui percentile alti. Accetta buchi di rosa per assicurarsi 2-3 nomi chiave.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.4,
        spilloverAdjacentTier: 0.4,
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
        maxInflationMultiplier: 2.0,
        baseInflationRate: 0.08,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.3,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.9,
      inflationTolerance: 0.85,
      maxOverpayRatio: 1.55,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.45,
      budgetElasticity: 0.8,
      varWeight: 0.7,
      teamStrengthWeight: 0.4,
      preferAlternatives: false,
      rebidTriggerPctAboveExpected: 0.3,
      targetTopTierCount: 4,
      budgetShareByRole: {
        P: 0.05,
        D: 0.15,
        C: 0.25,
        A: 0.55,
      },
    },
  },
  {
    id: "value_hunter",
    name: "Value Hunter",
    labelIt: "Cacciatore di valore",
    description: "Massimizza ESV/VR: predilige low-cost con alto projected_score/expected_price. Overpay quasi nullo; VAR e alternative low-cost pesano molto.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.25,
        spilloverAdjacentTier: 0.2,
        spilloverCrossRole: 0.0,
        minIndex: 0.55,
        maxIndex: 1.5,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.3,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.8,
        maxInflationMultiplier: 1.4,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.05,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.35,
      inflationTolerance: 0.25,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 2,
      allInProbability: 0.08,
      budgetElasticity: 0.25,
      varWeight: 0.75,
      teamStrengthWeight: 0.1,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.06,
      budgetShareByRole: {
        P: 0.07,
        D: 0.23,
        C: 0.3,
        A: 0.4,
      },
    },
  },
  {
    id: "inflation_fighter",
    name: "Inflation Fighter",
    labelIt: "Anti-inflazione",
    description: "Assume mercato caldo: inflazione alta nel baseline, EWMA reattivo, non insegue i picchi TOP. Preferisce MID con solidità e alternative.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.5,
        spilloverAdjacentTier: 0.3,
        spilloverCrossRole: 0.1,
        minIndex: 0.5,
        maxIndex: 2.0,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.35,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.55,
        maxInflationMultiplier: 2.0,
        baseInflationRate: 0.09,
        baselineParticipants: 6,
        teamStrengthMultiplier: 0.25,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.4,
      inflationTolerance: 0.2,
      maxOverpayRatio: 1.12,
      minResidualCreditsPerSlot: 2,
      allInProbability: 0.1,
      budgetElasticity: 0.3,
      varWeight: 0.4,
      teamStrengthWeight: 0.45,
      preferAlternatives: true,
      avoidTopTierEarly: true,
      rebidTriggerPctAboveExpected: 0.08,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.3,
        A: 0.38,
      },
    },
  },
  {
    id: "early_stars",
    name: "Early Stars",
    labelIt: "Stelle all'inizio",
    description: "Spende forte nella prima fase d'asta sui nomi chiave, poi si adatta. Alpha alto all'inizio concettuale; overpay consentito solo sui TOP.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.35,
        spilloverAdjacentTier: 0.3,
        spilloverCrossRole: 0.05,
        minIndex: 0.5,
        maxIndex: 1.9,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.4,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.65,
        maxInflationMultiplier: 1.8,
        baseInflationRate: 0.06,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.2,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.75,
      inflationTolerance: 0.65,
      maxOverpayRatio: 1.4,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.3,
      budgetElasticity: 0.55,
      varWeight: 0.5,
      teamStrengthWeight: 0.3,
      preferAlternatives: false,
      phaseBias: "early",
      rebidTriggerPctAboveExpected: 0.22,
      budgetShareByRole: {
        P: 0.05,
        D: 0.18,
        C: 0.27,
        A: 0.5,
      },
    },
  },
  {
    id: "late_sniper",
    name: "Late Sniper",
    labelIt: "Cecchino di fine asta",
    description: "Conserva budget e agisce a fine asta su residual value. Alpha moderato, overpay basso, alternative low-cost prioritizzate, all-in raro.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.28,
        spilloverAdjacentTier: 0.22,
        spilloverCrossRole: 0.0,
        minIndex: 0.55,
        maxIndex: 1.6,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.28,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.75,
        maxInflationMultiplier: 1.45,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.08,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.3,
      inflationTolerance: 0.35,
      maxOverpayRatio: 1.1,
      minResidualCreditsPerSlot: 3,
      allInProbability: 0.15,
      budgetElasticity: 0.35,
      varWeight: 0.6,
      teamStrengthWeight: 0.15,
      preferAlternatives: true,
      phaseBias: "late",
      rebidTriggerPctAboveExpected: 0.08,
      budgetShareByRole: {
        P: 0.08,
        D: 0.22,
        C: 0.3,
        A: 0.4,
      },
    },
  },
  {
    id: "youth_first",
    name: "Youth First",
    labelIt: "Giovani first",
    description: "Privilegia season_value e potenziale (giovani in rampa). Drift moderato; team strength meno rilevante; tollera minutaggio incerto se ESV alto.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.3,
        spilloverAdjacentTier: 0.2,
        spilloverCrossRole: 0.0,
        minIndex: 0.5,
        maxIndex: 1.7,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.35,
      },
      useInflationBaseline: false,
      inflationConfig: {
        inflationPercentileThreshold: 0.7,
        maxInflationMultiplier: 1.5,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.0,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.45,
      inflationTolerance: 0.4,
      maxOverpayRatio: 1.18,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.18,
      budgetElasticity: 0.45,
      varWeight: 0.55,
      teamStrengthWeight: 0.05,
      preferAlternatives: true,
      preferYoungPlayers: true,
      maxAgePreference: 23,
      rebidTriggerPctAboveExpected: 0.15,
      budgetShareByRole: {
        P: 0.06,
        D: 0.2,
        C: 0.32,
        A: 0.42,
      },
    },
  },
  {
    id: "safe_picks",
    name: "Safe Picks",
    labelIt: "Scelte sicure",
    description: "Titolari consolidati, bassa varianza, overpay minimo. Idealmente abbinato a giocatori con Pr alta e DV bassa (certezze di Fase 8).",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.22,
        spilloverAdjacentTier: 0.15,
        spilloverCrossRole: 0.0,
        minIndex: 0.6,
        maxIndex: 1.45,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.38,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.72,
        maxInflationMultiplier: 1.4,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.12,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: 0.65,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.28,
      inflationTolerance: 0.3,
      maxOverpayRatio: 1.12,
      minResidualCreditsPerSlot: 2,
      allInProbability: 0.06,
      budgetElasticity: 0.25,
      varWeight: 0.25,
      teamStrengthWeight: 0.2,
      preferAlternatives: true,
      preferHighStartProbability: true,
      minStartProbability: 0.65,
      rebidTriggerPctAboveExpected: 0.07,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.28,
        A: 0.4,
      },
    },
  },
  {
    id: "risk_lover",
    name: "Risk Lover",
    labelIt: "Amante del rischio",
    description: "Scommesse, svincolati, high ceiling. Season value, spillover cross-role, all-in frequente su outlier VAR.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.48,
        spilloverAdjacentTier: 0.35,
        spilloverCrossRole: 0.12,
        minIndex: 0.5,
        maxIndex: 2.0,
        tierThresholds: [0.3, 0.7],
      },
      alternativesConfig: {
        lowCostPercentile: 0.55,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.5,
        maxInflationMultiplier: 2.0,
        baseInflationRate: 0.08,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.15,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.92,
      inflationTolerance: 0.75,
      maxOverpayRatio: 1.5,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.5,
      budgetElasticity: 0.85,
      varWeight: 0.8,
      teamStrengthWeight: 0.15,
      preferAlternatives: false,
      preferHighVariance: true,
      rebidTriggerPctAboveExpected: 0.28,
      budgetShareByRole: {
        P: 0.04,
        D: 0.16,
        C: 0.28,
        A: 0.52,
      },
    },
  },
  {
    id: "mantra_optimized",
    name: "Mantra Optimized",
    labelIt: "Ottimizzato Mantra",
    description: "Pensato per leghe MANTRA: valorizza polivalenza (Num_Ruoli), flessibilità di modulo, alternative multi-ruolo. Budget share allineato ai 6 blocchi algoritmo v3.1.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.3,
        spilloverAdjacentTier: 0.25,
        spilloverCrossRole: 0.08,
        minIndex: 0.5,
        maxIndex: 1.8,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.4,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.7,
        maxInflationMultiplier: 1.6,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.15,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.55,
      inflationTolerance: 0.5,
      maxOverpayRatio: 1.22,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.15,
      budgetElasticity: 0.45,
      varWeight: 0.45,
      teamStrengthWeight: 0.2,
      preferAlternatives: true,
      preferMultiRole: true,
      minNumRoles: 2,
      rebidTriggerPctAboveExpected: 0.14,
      budgetShareByBlock: {
        Por: 0.06,
        Difesa_Pura: 0.14,
        Ibridi_Difensivi: 0.12,
        Centro_Nevralgico: 0.18,
        Linea_Fantasia: 0.18,
        Attacco: 0.32,
      },
      budgetShareByRole: {
        P: 0.06,
        D: 0.26,
        C: 0.3,
        A: 0.38,
      },
    },
  },
  {
    id: "depth_builder",
    name: "Depth Builder",
    labelIt: "Costruttore di profondit\u00e0",
    description: "Priorità a copertura ruoli e panchina utile piuttosto che a un singolo superstar. Budget più uniforme, residuale alto, alternative sempre considerate.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.25,
        spilloverAdjacentTier: 0.2,
        spilloverCrossRole: 0.0,
        minIndex: 0.55,
        maxIndex: 1.55,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.32,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.75,
        maxInflationMultiplier: 1.4,
        baseInflationRate: 0.04,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.08,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.35,
      inflationTolerance: 0.35,
      maxOverpayRatio: 1.12,
      minResidualCreditsPerSlot: 2,
      allInProbability: 0.05,
      budgetElasticity: 0.3,
      varWeight: 0.3,
      teamStrengthWeight: 0.15,
      preferAlternatives: true,
      maxTopTierCount: 2,
      rebidTriggerPctAboveExpected: 0.08,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.3,
        A: 0.38,
      },
    },
  },
  {
    id: "budget_saver",
    name: "Budget Saver",
    labelIt: "Risparmiatore",
    description: "Massimizza residuo e flessibilità di fine asta. Quasi zero overpay; credit reserve più stretta del minimo legale.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.18,
        spilloverAdjacentTier: 0.12,
        spilloverCrossRole: 0.0,
        minIndex: 0.65,
        maxIndex: 1.35,
        tierThresholds: [0.4, 0.8],
      },
      alternativesConfig: {
        lowCostPercentile: 0.25,
      },
      useInflationBaseline: false,
      inflationConfig: {
        inflationPercentileThreshold: 0.85,
        maxInflationMultiplier: 1.25,
        baseInflationRate: 0.03,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.0,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.15,
      inflationTolerance: 0.15,
      maxOverpayRatio: 1.05,
      minResidualCreditsPerSlot: 4,
      allInProbability: 0.02,
      budgetElasticity: 0.1,
      varWeight: 0.4,
      teamStrengthWeight: 0.05,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.03,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.3,
        A: 0.38,
      },
    },
  },
  {
    id: "adaptive_ai",
    name: "Adaptive AI",
    labelIt: "AI adattiva",
    description: "Profilo meta: EWMA reattivo, VAR e team strength bilanciati, valuation season, inflazione attiva. Pensato per essere aggiornato dinamicamente in base a price_index e residuo.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.38,
        spilloverAdjacentTier: 0.28,
        spilloverCrossRole: 0.06,
        minIndex: 0.5,
        maxIndex: 1.9,
        tierThresholds: [0.38, 0.78],
      },
      alternativesConfig: {
        lowCostPercentile: 0.4,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.65,
        maxInflationMultiplier: 1.75,
        baseInflationRate: 0.06,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.22,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.6,
      inflationTolerance: 0.55,
      maxOverpayRatio: 1.28,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.22,
      budgetElasticity: 0.6,
      varWeight: 0.65,
      teamStrengthWeight: 0.3,
      preferAlternatives: true,
      adaptive: true,
      adaptOn: [
        "price_index",
        "budget_residual",
        "role_fill_rate",
      ],
      rebidTriggerPctAboveExpected: 0.18,
      budgetShareByRole: {
        P: 0.06,
        D: 0.2,
        C: 0.3,
        A: 0.44,
      },
    },
  },
] as const;

export const AUCTION_PRESETS_BY_ID: ReadonlyMap<string, AuctionPreset> =
  new Map(AUCTION_PRESETS.map(p => [p.id, p]));

export function findAuctionPreset(id: string): AuctionPreset | undefined {
  return AUCTION_PRESETS_BY_ID.get(id);
}