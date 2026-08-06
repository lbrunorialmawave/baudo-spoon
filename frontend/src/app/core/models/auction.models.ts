// ── Domain primitives ───────────────────────────────────────────────────────

import { InflationConfig } from "./api.models";

/** Fantacalcio role code (CLASSIC). MANTRA uses string role codes. */
export type AuctionRole = 'P' | 'D' | 'C' | 'A';

/** Ruleset: CLASSIC (4 roles) or MANTRA (12 multi-slot roles). */
export type AuctionRuleset = 'CLASSIC' | 'MANTRA';

/** Price-drift tier classification. */
export type AuctionTier = 'LOW' | 'MID' | 'TOP';

/** All CLASSIC roles in display order. */
export const AUCTION_ROLES: readonly AuctionRole[] = ['P', 'D', 'C', 'A'] as const;

/** Default MANTRA role quotas (mirrors ml.optimizer.models.MANTRA_DEFAULT_QUOTAS). */
export const MANTRA_DEFAULT_QUOTAS: Readonly<Record<string, number>> = {
  Por: 3,
  Dc: 3, B: 2, Dd: 2, Ds: 1,
  E: 1, M: 2, C: 5,
  T: 1, W: 1, A: 2, Pc: 2,
} as const;

/** MANTRA role codes in stable display order. */
export const MANTRA_ROLES: readonly string[] = Object.keys(MANTRA_DEFAULT_QUOTAS);

/** All tiers from low to top. */
export const AUCTION_TIERS: readonly AuctionTier[] = ['LOW', 'MID', 'TOP'] as const;

// ── Configuration ──────────────────────────────────────────────────────────

/** EWMA + spillover coefficients for live price drift. */
export interface MarketDriftConfig {
  alpha: number;
  spilloverAdjacentTier: number;
  spilloverCrossRole: number;
  minIndex: number;
  maxIndex: number;
  /** Tuple [lowThreshold, topThreshold], each in [0, 1]. */
  tierThresholds: readonly [number, number];
}

/** Low-cost alternative heuristic. */
export interface AlternativesConfig {
  lowCostPercentile: number;
}

/** Score metric used by VarEngine and the optimizer objective. */
export type ValuationMode = 'PER_MATCH_RATING' | 'SEASON_VALUE';

/** Replacement-level strategy used by VarEngine. */
export type ReplacementMethod = 'percentile' | 'roster_depth';

/** Auction market + roster quotas. */
export interface AuctionConfig {
  numParticipants: number;
  /** Map of role -> quota (3P/8D/8C/6A by default for CLASSIC). */
  roleQuotas: Partial<Record<string, number>>;
  /** Ruleset: CLASSIC (default) or MANTRA. */
  ruleset?: AuctionRuleset;
  marketDriftConfig: MarketDriftConfig;
  alternativesConfig: AlternativesConfig;
  useInflationBaseline: boolean;
  /**
   * Weight in [0, 1] of the fpIbrido (MANTRA-ibrido) signal in VarEngine.
   * 0 = disabled (default). Same shape as optimizer hybridBlend.
   */
  hybridBlend?: number;
  /**
   * Full inflation config (percentile threshold, max multiplier, base
   * rate, baseline participants, Club Elo weight). When `useInflationBaseline`
   * is `true` and this object is sent, the backend uses its values; when
   * omitted, the backend uses its own defaults (mirrors Optimizer).
   * Sending this object while `useInflationBaseline` is `false` is
   * silently ignored by the backend.
   */
  inflationConfig?: Partial<InflationConfig> | null;
  /**
   * Pre-filter per il ranking VAR: scarta i giocatori con
   * `start_probability` < soglia PRIMA del ranking. `null` (default)
   * = nessun filtro, allineato al default Pydantic lato backend.
   */
  minStartProbability?: number | null;
  /**
   * Metodo di calcolo del replacement level per VAR/ESV.
   * 'percentile' (default backend) = bottom-N% per ruolo,
   * 'roster_depth' = quota di rosa per ruolo.
   */
  replacementMethod?: ReplacementMethod;
  /** Budget the quotation file is calibrated on (historical baseline). */
  referenceBudget: number;
  /** Budget per team for the current auction session. */
  budgetInitial: number;
  /** Score metric: PER_MATCH_RATING (default) or SEASON_VALUE. */
  valuationMode?: ValuationMode;
}

// ── Setup payloads ─────────────────────────────────────────────────────────

/** Initial setup for one participant. */
export interface AuctionParticipantSetup {
  participantId: string;
  displayName: string;
  budgetInitial: number;
}

/** Player available in the auction pool. */
export interface AuctionPlayer {
  playerId: string;
  name: string;
  role: AuctionRole | string;
  realTeam: string;
  cost: number;
  projectedScore: number;
  /** MANTRA only: role codes this player can fill. */
  eligibleRoles?: string[];
}

// ── Init / lifecycle ──────────────────────────────────────────────────────

/**
 * Request body for `POST /auction/init`.
 *
 * - `seasonStart` is **required**: the backend uses it to look up
 *   quotations and ML predictions from the DB.
 * - `playerPool` is **optional**: when omitted, the backend builds the
 *   pool from the DB (same pattern as `OptimizationRequest.poolOverride`).
 *   Pass it explicitly only for tests, custom fixtures, or sessions that
 *   need a restricted pool (e.g. leftovers from a previous auction).
 */
export interface InitializeAuctionRequest {
  seasonStart: number;
  participants: AuctionParticipantSetup[];
  config: AuctionConfig;
  playerPool?: AuctionPlayer[];
}

export interface InitializeAuctionResponse {
  sessionId: string;
}

// ── Record assignment ─────────────────────────────────────────────────────

export interface RecordAssignmentRequest {
  playerId: string;
  winnerParticipantId: string;
  finalPrice: number;
  /** MANTRA only: explicit slot filled (auto-picked if omitted). */
  assignedSlot?: string | null;
}

/**
 * Outcome of a `record` call.
 *
 * The backend returns HTTP 200 with `success=false` for *validation*
 * rejections (unknown player/participant, role full, credit reserve
 * violation). Callers should branch on `success` rather than on status
 * code; HTTP 4xx/5xx are reserved for server-side faults.
 */
export interface RecordAssignmentResponse {
  success: boolean;
  sequenceNumber?: number;
  priceIndexAfter?: number;
  rejectionCode?: string;
  rejectionReason?: string;
}

// ── Read models (summary) ─────────────────────────────────────────────────

/** Lightweight player view nested in participant + assignment records. */
export interface AuctionPlayerSummary {
  playerId: string;
  name: string;
  realTeam: string;
  role: AuctionRole | string;
  cost: number;
  projectedScore: number;
  eligibleRoles?: string[] | null;
}

export interface AuctionParticipantState {
  participantId: string;
  displayName: string;
  budgetResidual: number;
  squad: AuctionPlayerSummary[];
  roleBreakdown: Partial<Record<string, number>>;
}

export interface AssignmentRecord {
  sequenceNumber: number;
  player: AuctionPlayerSummary;
  winnerParticipantId: string;
  finalPrice: number;
  role: AuctionRole | string;
  tier: AuctionTier;
  priceIndexBefore: number;
  priceIndexAfter: number;
  /** MANTRA slot filled; equals role under CLASSIC. */
  assignedSlot?: string | null;
}

export interface AuctionSummary {
  participants: AuctionParticipantState[];
  assignments: AssignmentRecord[];
  /** Nested map: role -> tier -> price index. */
  priceIndex: Partial<Record<string, Partial<Record<AuctionTier, number>>>>;
  /**
   * WS3 #1: participantId → P(complete roster | residual budget + slots).
   * Absent on older backends / pre-WS3 sessions.
   */
  completionProbability?: Record<string, number> | null;
}

// ── Live projection ───────────────────────────────────────────────────────

export interface ProjectionResponse {
  playerId: string;
  expectedPrice: number;
  tier: AuctionTier;
}

export interface AlternativesRequest {
  config?: AlternativesConfig | null;
  /** When set, backend computes maxAffordableBid for this manager. */
  participantId?: string | null;
  /** BALANCED | SUPER_DEFENSIVE | SUPER_OFFENSIVE | MIXED → strategyPriceCap. */
  strategyName?: string | null;
}

export interface AlternativesResponse {
  targetPlayerId: string;
  lowCostAlternative: AuctionPlayerSummary | null;
  closestAlternative: AuctionPlayerSummary | null;
  reasonIfNone: string | null;
  /** WS3 #3: mini-Pareto diversified candidates. */
  diversifiedAlternatives?: AuctionPlayerSummary[];
  /** WS3 #4: credit-reserve max bid for the requested participant. */
  maxAffordableBid?: number | null;
  /** WS3 #5: strategy-weighted price threshold. */
  strategyPriceCap?: number | null;
}

// ── Persistence (save / resume an auction) ────────────────────────────────

/** Server-serialized state payload; structure is backend-owned. */
export interface SerializedAuctionStateResponse {
  payload: Readonly<Record<string, unknown>>;
}

export interface DeserializeAuctionRequest {
  payload: Readonly<Record<string, unknown>>;
}

// ── VAR / ESV ranking ────────────────────────────────────────────────────────

export interface VarRankingItem {
  playerId: string;
  name: string;
  role: AuctionRole | string;
  projectedScore: number;
  varScore: number;
  expectedPrice: number;
  esv: number;
  calibrated: boolean;
  buySignal: boolean;
  seasonValue?: number | null;
  startProbability?: number | null;
}

export interface VarRankingResponse {
  sessionId: string;
  items: VarRankingItem[];
  usingLivePrices: boolean;
}
