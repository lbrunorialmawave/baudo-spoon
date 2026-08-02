/**
 * Auction strategy presets — single source of truth for the Auction setup UI.
 *
 * Recalibrated against Quotazioni Fantacalcio 2025/26:
 * - Qt.A mean≈8, median≈7; elite band ≥18 (≈30 players); solid band ≥8 (≈248)
 * - Higher Qt.A ↔ higher FVM reliability (r≈0.76)
 * - Tuning prioritises coherence: conservative protects residual & avoids noise;
 *   aggressive tolerates overpay on elite Qt.A names; value/late profiles
 *   lean on mid-tier (Qt.A 5–12) where projected_score/price is best.
 *
 * @see AuctionConfig / InitializeAuctionRequest in core/models/auction.models.ts
 */
import { AuctionConfig } from '../models/auction.models';
import { DEFAULT_CLASSIC_ROLE_QUOTAS } from './shared-presets';

/**
 * Identity function used purely for contextual type-checking.
 * Prefer this over `obj as AuctionConfig` in every new preset.
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
    description:
      "Evita overpay, protegge il residuo, predilige alternative low-cost e inflazione bassa. Filtra il rumore (Qt.A bassi). Ideale se sei ultimo di budget o temi di restare scoperto a fine asta.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.18,
        spilloverAdjacentTier: 0.12,
        spilloverCrossRole: 0.0,
        minIndex: 0.65,
        maxIndex: 1.35,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.30,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.78,
        maxInflationMultiplier: 1.30,
        baseInflationRate: 0.035,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.04,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: 0.55,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.22,
      inflationTolerance: 0.25,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 3,
      allInProbability: 0.04,
      budgetElasticity: 0.18,
      varWeight: 0.18,
      teamStrengthWeight: 0.08,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.04,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.28,
        A: 0.40,
      },
    },
  },
  {
    id: "balanced",
    name: "Balanced",
    labelIt: "Bilanciato",
    description:
      "Profilo neutro: EWMA standard, inflazione moderata, alternative standard. Punto di partenza consigliato per la maggior parte delle leghe a 8.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.30,
        spilloverAdjacentTier: 0.25,
        spilloverCrossRole: 0.0,
        minIndex: 0.55,
        maxIndex: 1.70,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.38,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.70,
        maxInflationMultiplier: 1.55,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.10,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.50,
      inflationTolerance: 0.50,
      maxOverpayRatio: 1.18,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.12,
      budgetElasticity: 0.40,
      varWeight: 0.35,
      teamStrengthWeight: 0.18,
      preferAlternatives: true,
      rebidTriggerPctAboveExpected: 0.12,
      budgetShareByRole: {
        P: 0.07,
        D: 0.21,
        C: 0.30,
        A: 0.42,
      },
    },
  },
  {
    id: "aggressive",
    name: "Aggressive",
    labelIt: "Aggressivo",
    description:
      "Insegue i top (Qt.A elite ≥18), tollera overpay e inflazione alta, spillover più reattivo. Rischia residuo basso ma punta a massimizzare il ceiling della rosa.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.42,
        spilloverAdjacentTier: 0.35,
        spilloverCrossRole: 0.05,
        minIndex: 0.50,
        maxIndex: 2.00,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.48,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.58,
        maxInflationMultiplier: 1.85,
        baseInflationRate: 0.07,
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
      aggressiveness: 0.82,
      inflationTolerance: 0.78,
      maxOverpayRatio: 1.42,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.32,
      budgetElasticity: 0.68,
      varWeight: 0.55,
      teamStrengthWeight: 0.32,
      preferAlternatives: false,
      rebidTriggerPctAboveExpected: 0.24,
      budgetShareByRole: {
        P: 0.05,
        D: 0.17,
        C: 0.28,
        A: 0.50,
      },
    },
  },
  {
    id: "top_player_hunter",
    name: "Top Player Hunter",
    labelIt: "Cacciatore di top",
    description:
      "Concentra budget sui TOP tier (Qt.A ≥18 / FVM alto); spillover alto sui top, all-in più probabile. Accetta buchi di rosa per 2–4 nomi chiave.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.40,
        spilloverAdjacentTier: 0.40,
        spilloverCrossRole: 0.08,
        minIndex: 0.50,
        maxIndex: 2.00,
        tierThresholds: [0.30, 0.70],
      },
      alternativesConfig: {
        lowCostPercentile: 0.45,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.52,
        maxInflationMultiplier: 2.00,
        baseInflationRate: 0.08,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.28,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.90,
      inflationTolerance: 0.85,
      maxOverpayRatio: 1.55,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.42,
      budgetElasticity: 0.78,
      varWeight: 0.70,
      teamStrengthWeight: 0.38,
      preferAlternatives: false,
      rebidTriggerPctAboveExpected: 0.28,
      targetTopTierCount: 4,
      budgetShareByRole: {
        P: 0.04,
        D: 0.14,
        C: 0.26,
        A: 0.56,
      },
    },
  },
  {
    id: "value_hunter",
    name: "Value Hunter",
    labelIt: "Cacciatore di valore",
    description:
      "Massimizza ESV/VR nella fascia Qt.A 5–12 (solid mid-tier). Overpay quasi nullo; VAR e alternative low-cost pesano molto. Evita il rumore Qt.A≤3.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.22,
        spilloverAdjacentTier: 0.18,
        spilloverCrossRole: 0.0,
        minIndex: 0.58,
        maxIndex: 1.45,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.28,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.80,
        maxInflationMultiplier: 1.35,
        baseInflationRate: 0.035,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.04,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.32,
      inflationTolerance: 0.22,
      maxOverpayRatio: 1.06,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.06,
      budgetElasticity: 0.22,
      varWeight: 0.78,
      teamStrengthWeight: 0.08,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.05,
      budgetShareByRole: {
        P: 0.07,
        D: 0.23,
        C: 0.31,
        A: 0.39,
      },
    },
  },
  {
    id: "inflation_fighter",
    name: "Inflation Fighter",
    labelIt: "Anti-inflazione",
    description:
      "Assume mercato caldo: inflazione alta nel baseline, EWMA reattivo, non insegue i picchi TOP (Qt.A elite). Preferisce MID solidi (Qt.A 6–12) e alternative.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.48,
        spilloverAdjacentTier: 0.30,
        spilloverCrossRole: 0.08,
        minIndex: 0.50,
        maxIndex: 2.00,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.32,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.52,
        maxInflationMultiplier: 2.00,
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
      aggressiveness: 0.38,
      inflationTolerance: 0.18,
      maxOverpayRatio: 1.10,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.08,
      budgetElasticity: 0.28,
      varWeight: 0.42,
      teamStrengthWeight: 0.42,
      preferAlternatives: true,
      avoidTopTierEarly: true,
      rebidTriggerPctAboveExpected: 0.07,
      budgetShareByRole: {
        P: 0.08,
        D: 0.25,
        C: 0.30,
        A: 0.37,
      },
    },
  },
  {
    id: "early_stars",
    name: "Early Stars",
    labelIt: "Stelle all'inizio",
    description:
      "Spende forte nella prima fase sui nomi chiave (elite Qt.A), poi si adatta. Overpay consentito solo sui TOP; dopo fase 1 diventa più selettivo.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.36,
        spilloverAdjacentTier: 0.30,
        spilloverCrossRole: 0.05,
        minIndex: 0.50,
        maxIndex: 1.90,
        tierThresholds: [0.35, 0.75],
      },
      alternativesConfig: {
        lowCostPercentile: 0.40,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.62,
        maxInflationMultiplier: 1.75,
        baseInflationRate: 0.06,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.20,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.72,
      inflationTolerance: 0.62,
      maxOverpayRatio: 1.38,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.28,
      budgetElasticity: 0.55,
      varWeight: 0.50,
      teamStrengthWeight: 0.28,
      preferAlternatives: false,
      phaseBias: "early",
      rebidTriggerPctAboveExpected: 0.20,
      budgetShareByRole: {
        P: 0.05,
        D: 0.17,
        C: 0.28,
        A: 0.50,
      },
    },
  },
  {
    id: "late_sniper",
    name: "Late Sniper",
    labelIt: "Cecchino di fine asta",
    description:
      "Conserva budget e agisce a fine asta su residual value nella fascia Qt.A 5–11. Alpha moderato, overpay basso, alternative low-cost prioritizzate.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.25,
        spilloverAdjacentTier: 0.20,
        spilloverCrossRole: 0.0,
        minIndex: 0.58,
        maxIndex: 1.55,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.26,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.76,
        maxInflationMultiplier: 1.40,
        baseInflationRate: 0.035,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.06,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.28,
      inflationTolerance: 0.32,
      maxOverpayRatio: 1.08,
      minResidualCreditsPerSlot: 3.5,
      allInProbability: 0.12,
      budgetElasticity: 0.32,
      varWeight: 0.62,
      teamStrengthWeight: 0.12,
      preferAlternatives: true,
      phaseBias: "late",
      rebidTriggerPctAboveExpected: 0.06,
      budgetShareByRole: {
        P: 0.08,
        D: 0.23,
        C: 0.30,
        A: 0.39,
      },
    },
  },
  {
    id: "youth_first",
    name: "Youth First",
    labelIt: "Giovani first",
    description:
      "Privilegia season_value e potenziale (giovani in rampa). Drift moderato; team strength meno rilevante; tollera minutaggio incerto se ESV alto.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.28,
        spilloverAdjacentTier: 0.18,
        spilloverCrossRole: 0.0,
        minIndex: 0.52,
        maxIndex: 1.65,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.35,
      },
      useInflationBaseline: false,
      inflationConfig: {
        inflationPercentileThreshold: 0.70,
        maxInflationMultiplier: 1.45,
        baseInflationRate: 0.045,
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
      inflationTolerance: 0.38,
      maxOverpayRatio: 1.16,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.16,
      budgetElasticity: 0.42,
      varWeight: 0.55,
      teamStrengthWeight: 0.04,
      preferAlternatives: true,
      preferYoungPlayers: true,
      maxAgePreference: 23,
      rebidTriggerPctAboveExpected: 0.14,
      budgetShareByRole: {
        P: 0.06,
        D: 0.20,
        C: 0.32,
        A: 0.42,
      },
    },
  },
  {
    id: "safe_picks",
    name: "Safe Picks",
    labelIt: "Scelte sicure",
    description:
      "Titolari consolidati (alta Pr, Qt.A solidi ≥8), bassa varianza, overpay minimo. Ideale per chi vuole certezze e floor elevato.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.20,
        spilloverAdjacentTier: 0.14,
        spilloverCrossRole: 0.0,
        minIndex: 0.62,
        maxIndex: 1.40,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.36,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.74,
        maxInflationMultiplier: 1.38,
        baseInflationRate: 0.035,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.10,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: 0.70,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.25,
      inflationTolerance: 0.28,
      maxOverpayRatio: 1.10,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.05,
      budgetElasticity: 0.22,
      varWeight: 0.22,
      teamStrengthWeight: 0.18,
      preferAlternatives: true,
      preferHighStartProbability: true,
      minStartProbability: 0.70,
      rebidTriggerPctAboveExpected: 0.06,
      budgetShareByRole: {
        P: 0.08,
        D: 0.25,
        C: 0.28,
        A: 0.39,
      },
    },
  },
  {
    id: "risk_lover",
    name: "Risk Lover",
    labelIt: "Amante del rischio",
    description:
      "Scommesse, svincolati, high ceiling, outlier VAR. Season value, spillover cross-role, all-in frequente. Accetta Qt.A bassi se il potenziale è alto.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.48,
        spilloverAdjacentTier: 0.35,
        spilloverCrossRole: 0.12,
        minIndex: 0.50,
        maxIndex: 2.00,
        tierThresholds: [0.28, 0.68],
      },
      alternativesConfig: {
        lowCostPercentile: 0.55,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.48,
        maxInflationMultiplier: 2.00,
        baseInflationRate: 0.08,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.14,
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
      maxOverpayRatio: 1.50,
      minResidualCreditsPerSlot: 1,
      allInProbability: 0.48,
      budgetElasticity: 0.85,
      varWeight: 0.82,
      teamStrengthWeight: 0.12,
      preferAlternatives: false,
      preferHighVariance: true,
      rebidTriggerPctAboveExpected: 0.28,
      budgetShareByRole: {
        P: 0.04,
        D: 0.15,
        C: 0.28,
        A: 0.53,
      },
    },
  },
  {
    id: "mantra_optimized",
    name: "Mantra Optimized",
    labelIt: "Ottimizzato Mantra",
    description:
      "Pensato per leghe MANTRA: valorizza polivalenza (Num_Ruoli), flessibilità di modulo, alternative multi-ruolo. Budget share allineato ai 6 blocchi algoritmo v3.1.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.30,
        spilloverAdjacentTier: 0.25,
        spilloverCrossRole: 0.10,
        minIndex: 0.52,
        maxIndex: 1.75,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.38,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.68,
        maxInflationMultiplier: 1.55,
        baseInflationRate: 0.05,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.14,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.52,
      inflationTolerance: 0.48,
      maxOverpayRatio: 1.20,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.14,
      budgetElasticity: 0.45,
      varWeight: 0.45,
      teamStrengthWeight: 0.18,
      preferAlternatives: true,
      preferMultiRole: true,
      minNumRoles: 2,
      rebidTriggerPctAboveExpected: 0.13,
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
        C: 0.30,
        A: 0.38,
      },
    },
  },
  {
    id: "depth_builder",
    name: "Depth Builder",
    labelIt: "Costruttore di profondità",
    description:
      "Priorità a copertura ruoli e panchina utile (Qt.A usable 5–10) piuttosto che a un singolo superstar. Budget più uniforme, residuale alto.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.24,
        spilloverAdjacentTier: 0.18,
        spilloverCrossRole: 0.0,
        minIndex: 0.58,
        maxIndex: 1.50,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.30,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.76,
        maxInflationMultiplier: 1.38,
        baseInflationRate: 0.035,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.06,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "PER_MATCH_RATING",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.32,
      inflationTolerance: 0.32,
      maxOverpayRatio: 1.10,
      minResidualCreditsPerSlot: 2.5,
      allInProbability: 0.04,
      budgetElasticity: 0.28,
      varWeight: 0.30,
      teamStrengthWeight: 0.12,
      preferAlternatives: true,
      maxTopTierCount: 2,
      rebidTriggerPctAboveExpected: 0.07,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.30,
        A: 0.38,
      },
    },
  },
  {
    id: "budget_saver",
    name: "Budget Saver",
    labelIt: "Risparmiatore",
    description:
      "Massimizza residuo e flessibilità di fine asta. Quasi zero overpay; credit reserve stretta. Filtra il rumore e punta su low-cost solidi.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.15,
        spilloverAdjacentTier: 0.10,
        spilloverCrossRole: 0.0,
        minIndex: 0.68,
        maxIndex: 1.30,
        tierThresholds: [0.40, 0.80],
      },
      alternativesConfig: {
        lowCostPercentile: 0.22,
      },
      useInflationBaseline: false,
      inflationConfig: {
        inflationPercentileThreshold: 0.85,
        maxInflationMultiplier: 1.22,
        baseInflationRate: 0.025,
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
      aggressiveness: 0.12,
      inflationTolerance: 0.12,
      maxOverpayRatio: 1.04,
      minResidualCreditsPerSlot: 4.5,
      allInProbability: 0.02,
      budgetElasticity: 0.08,
      varWeight: 0.38,
      teamStrengthWeight: 0.04,
      preferAlternatives: true,
      preferLowCostAlternative: true,
      rebidTriggerPctAboveExpected: 0.03,
      budgetShareByRole: {
        P: 0.08,
        D: 0.24,
        C: 0.30,
        A: 0.38,
      },
    },
  },
  {
    id: "adaptive_ai",
    name: "Adaptive AI",
    labelIt: "AI adattiva",
    description:
      "Profilo meta: EWMA reattivo, VAR e team strength bilanciati, valuation season, inflazione attiva. Si aggiorna dinamicamente su price_index e residuo.",
    config: defineAuctionConfig({
      numParticipants: 8,
      roleQuotas: DEFAULT_CLASSIC_ROLE_QUOTAS,
      marketDriftConfig: {
        alpha: 0.36,
        spilloverAdjacentTier: 0.28,
        spilloverCrossRole: 0.06,
        minIndex: 0.50,
        maxIndex: 1.85,
        tierThresholds: [0.38, 0.78],
      },
      alternativesConfig: {
        lowCostPercentile: 0.38,
      },
      useInflationBaseline: true,
      inflationConfig: {
        inflationPercentileThreshold: 0.64,
        maxInflationMultiplier: 1.70,
        baseInflationRate: 0.06,
        baselineParticipants: 8,
        teamStrengthMultiplier: 0.20,
      },
      referenceBudget: 300,
      budgetInitial: 300,
      valuationMode: "SEASON_VALUE",
      minStartProbability: null,
      replacementMethod: "percentile",
    }),
    policy: {
      aggressiveness: 0.58,
      inflationTolerance: 0.52,
      maxOverpayRatio: 1.25,
      minResidualCreditsPerSlot: 1.5,
      allInProbability: 0.20,
      budgetElasticity: 0.58,
      varWeight: 0.62,
      teamStrengthWeight: 0.28,
      preferAlternatives: true,
      adaptive: true,
      adaptOn: [
        "price_index",
        "budget_residual",
        "role_fill_rate",
      ],
      rebidTriggerPctAboveExpected: 0.16,
      budgetShareByRole: {
        P: 0.06,
        D: 0.20,
        C: 0.30,
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
