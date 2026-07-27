// ── Domain primitives ───────────────────────────────────────────────────────

/** Fantacalcio role code. */
export type AuctionRole = 'P' | 'D' | 'C' | 'A';

/** Price-drift tier classification. */
export type AuctionTier = 'LOW' | 'MID' | 'TOP';

/** All roles in display order. */
export const AUCTION_ROLES: readonly AuctionRole[] = ['P', 'D', 'C', 'A'] as const;

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

/** Auction market + roster quotas. */
export interface AuctionConfig {
  numParticipants: number;
  /** Map of role -> quota (3P/8D/8C/6A by default). */
  roleQuotas: Partial<Record<AuctionRole, number>>;
  marketDriftConfig: MarketDriftConfig;
  alternativesConfig: AlternativesConfig;
  useInflationBaseline: boolean;
  /** Budget the quotation file is calibrated on (historical baseline). */
  referenceBudget: number;
  /** Budget per team for the current auction session. */
  budgetInitial: number;
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
  role: AuctionRole;
  realTeam: string;
  cost: number;
  projectedScore: number;
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
  role: AuctionRole;
  cost: number;
  projectedScore: number;
}

export interface AuctionParticipantState {
  participantId: string;
  displayName: string;
  budgetResidual: number;
  squad: AuctionPlayerSummary[];
  roleBreakdown: Partial<Record<AuctionRole, number>>;
}

export interface AssignmentRecord {
  sequenceNumber: number;
  player: AuctionPlayerSummary;
  winnerParticipantId: string;
  finalPrice: number;
  role: AuctionRole;
  tier: AuctionTier;
  priceIndexBefore: number;
  priceIndexAfter: number;
}

export interface AuctionSummary {
  participants: AuctionParticipantState[];
  assignments: AssignmentRecord[];
  /** Nested map: role -> tier -> price index. */
  priceIndex: Partial<Record<AuctionRole, Partial<Record<AuctionTier, number>>>>;
}

// ── Live projection ───────────────────────────────────────────────────────

export interface ProjectionResponse {
  playerId: string;
  expectedPrice: number;
  tier: AuctionTier;
}

export interface AlternativesRequest {
  config?: AlternativesConfig | null;
}

export interface AlternativesResponse {
  targetPlayerId: string;
  lowCostAlternative: AuctionPlayerSummary | null;
  closestAlternative: AuctionPlayerSummary | null;
  reasonIfNone: string | null;
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
  role: AuctionRole;
  projectedScore: number;
  varScore: number;
  expectedPrice: number;
  esv: number;
  calibrated: boolean;
  buySignal: boolean;
}

export interface VarRankingResponse {
  sessionId: string;
  items: VarRankingItem[];
  usingLivePrices: boolean;
}
