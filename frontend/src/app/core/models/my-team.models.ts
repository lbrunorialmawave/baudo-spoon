/** Models for La Mia Squadra (roster import, lineup, trades) — camelCase API. */

export type Ruleset = 'MANTRA' | 'CLASSIC';

export interface RosterTeamCard {
  sheetName: string;
  teamName: string;
  playerCount: number;
  totalSpent: number;
  isEmpty: boolean;
  matchRate: number;
}

export interface RosterMatchQuality {
  totalPlayers: number;
  auto: number;
  provisional: number;
  unmatched: number;
  matchRate: number;
}

export interface RosterImportResponse {
  contextId: string;
  sourceFilename?: string | null;
  quality: RosterMatchQuality;
  teams: RosterTeamCard[];
  divisions: string[];
  expiresInSeconds: number;
}

export interface RosterClaimResponse {
  contextId: string;
  userTeamKey: string;
  teamName: string;
  sheetName: string;
  playerCount: number;
  totalSpent: number;
  matchRate: number;
}

export interface RosterPlayer {
  nameRaw: string;
  nameClean: string;
  cost: number;
  status: string;
  score: number;
  needsReview: boolean;
  fantacalcioId?: number | null;
  catalogName?: string | null;
  catalogTeam?: string | null;
  roleClassic?: string | null;
  rolesMantra: string[];
}

export interface RosterDetailResponse {
  contextId: string;
  sheetName: string;
  teamName: string;
  totalSpent: number;
  matchRate: number;
  players: RosterPlayer[];
}

export interface LineupOptimizeRequest {
  contextId: string;
  sheetName: string;
  teamName: string;
  matchday?: number | null;
  opponentSheetName?: string | null;
  opponentTeamName?: string | null;
  ruleset?: string;
  candidateFormations?: string[] | null;
  minStarterProb?: number;
}

export interface SlotAssignment {
  slotLabel: string;
  slotRoles: string[];
  playerId: string;
  playerName: string;
  expectedScore: number;
  starterProbability: number;
  breakdownNote?: string;
}

export interface FormationAlternative {
  formation: string;
  feasible: boolean;
  scoreTotale: number;
  reason?: string;
}

export interface LineupOptimizeResponse {
  contextId: string;
  teamName: string;
  sheetName: string;
  matchday?: number | null;
  chosenFormation?: string | null;
  scoreTotale?: number | null;
  startingXi: SlotAssignment[];
  bench: SlotAssignment[];
  alternativesConsidered: FormationAlternative[];
  opponentHeadToHead?: Record<string, unknown> | null;
  enrichment?: Record<string, unknown> | null;
  notes: string[];
}

export interface TradesDashboardRequest {
  contextId: string;
  sheetName: string;
  teamName: string;
  formationPrefs?: string[];
  hardExclusionThreshold?: number;
}

export interface TradesDashboardResponse {
  contextId: string;
  teamName: string;
  sheetName: string;
  formationPrefs: string[];
  coverageByFormation: Record<string, boolean>;
  coverageMatrix: Array<{
    formation: string;
    slotLabel: string;
    status: string;
    missing: number;
  }>;
  tradeOutCandidates: Array<{
    player: {
      playerId: string;
      name: string;
      roles: string[];
      cost: number;
      currentValue?: number | null;
      fpCorr: number;
      teamSerieA?: string;
    };
    retentionScore: number;
    surplusRoles: string[];
    rationale: string;
  }>;
  tradeInTargets: Array<{
    playerId: string;
    name: string;
    coversSlots: string[];
    fpCorr: number;
    estimatedCost: number;
    roles: string[];
  }>;
  excludedTopPerformers: Array<{
    player: { playerId: string; name: string; fpCorr: number; teamSerieA?: string };
    retentionScore: number;
    reason: string;
  }>;
  notes: string[];
}


export interface TradeLeg {
  playerId: string;
  originalPurchasePrice: number;
  currentValue: number;
}

export interface TradeExecuteRequest {
  contextId: string;
  sheetName: string;
  fromTeamName: string;
  toTeamName: string;
  give: TradeLeg[];
  receive: TradeLeg[];
  creditPenaltyEnabled?: boolean;
  decayStepPercent?: number;
  floorPercent?: number;
}

export interface TradeExecuteLegResult {
  playerId: string;
  direction: string;
  valueBefore: number;
  valueAfter: number;
  originalPurchasePrice: number;
  penaltyApplied: boolean;
}

export interface TradeExecuteResponse {
  transferId: string;
  contextId: string;
  sheetName: string;
  fromTeamName: string;
  toTeamName: string;
  legs: TradeExecuteLegResult[];
  creditPenaltyEnabled: boolean;
  recordedAt: number;
  notes?: string[];
}
