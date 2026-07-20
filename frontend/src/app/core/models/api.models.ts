// ── Generic pagination (DB-backed endpoints, snake_case fields) ──
export interface PaginatedResponse<T> {
  total: number;
  page: number;
  size: number;
  items: T[];
}

// ── Clustering ─────────────────────────────────────────────
export interface ClusteringStats {
  nClusters: number;
  silhouette: number | null;
  inertia: number | null;
  pcaExplainedVariance: number[] | null;
}

export interface PlayerCluster {
  playerName: string;
  playerFotmobId: number | null;
  teamName: string | null;
  canonicalRole: string | null;
  clusterId: number;
  pca0: number | null;
  pca1: number | null;
  predictedFantavoto: number | null;
}

export interface ClusteringResponse {
  clusteringStats: ClusteringStats;
  total: number;
  page: number;
  size: number;
  items: PlayerCluster[];
}

// ── Alternatives ───────────────────────────────────────────
export interface LowCostAlternative {
  topPlayerId: number | null;
  topPlayerName: string;
  topPlayerTeam: string | null;
  topPlayerFantavoto: number | null;
  altPlayerId: number | null;
  altPlayerName: string;
  altPlayerTeam: string | null;
  altPlayerFantavoto: number | null;
  clusterId: number;
  distance: number;
}

export interface AlternativesResponse {
  clusteringStats: ClusteringStats;
  playerClusters: PlayerCluster[];
  lowCostRecommendations: LowCostAlternative[];
}

// ── Predictions ────────────────────────────────────────────
export interface PlayerPrediction {
  playerName: string;
  playerFotmobId: number | null;
  teamName: string | null;
  canonicalRole: string | null;
  season: string | null;
  fantavotoMedio: number | null;  // actual (from training data)
  predicted: number;              // model prediction
}

export interface ModelComparison {
  model: string;
  rmse: number;
  mae: number;
  r2: number;
}

export interface NextSeasonPrediction {
  playerName: string;
  playerFotmobId: number | null;
  predictedNextFantavoto: number;
}

export interface PredictionsResponse {
  runId: string
  bestModel: string
  rolePartitioned: boolean
  modelComparison: ModelComparison[]
  total: number
  page: number
  size: number
  items: PlayerPrediction[]
}

// ── Optimizer ──────────────────────────────────────────────
export interface FormationConfig {
  label: string;
  defenders: number;
  midfielders: number;
  forwards: number;
}

export interface InflationConfig {
  inflationPercentileThreshold: number;
  maxInflationMultiplier: number;
  baseInflationRate: number;
  baselineParticipants: number;
}

export interface OptimizationRequest {
  seasonStart: number;
  budget?: number;
  numParticipants?: number;
  minQtA?: number;
  minDistinctTeams?: number;
  maxPlayersPerTeam?: number;
  bigTeams?: string[];
  bigTeamsCap?: number;
  formations?: FormationConfig[];
  inflationConfig?: Partial<InflationConfig>;
  solverTimeoutSeconds?: number;
  maxSinglePlayerBudgetShare?: number;
  mustInclude?: string[];
  exclude?: string[];
  ruleset?: 'CLASSIC' | 'MANTRA';
  mantraRoleQuotas?: Record<string, number> | null;
  preferredFormation?: FormationConfig | null;
  riskAversion?: number;
  strategyNames?: string[] | null;
}

export interface SquadPlayer {
  playerId: string;
  name: string;
  role: string;
  realTeam: string;
  cost: number;
  projectedScore: number;
  effectiveCost: number;
  predictionStd?: number | null;
}

export interface OptimizationResult {
  strategyName: string;
  status: string;
  squad: SquadPlayer[];
  totalNominalCost: number;
  totalEffectiveCost: number;
  totalProjectedScore: number;
  budgetResidual: number;
  roleBreakdown: Record<string, number>;
  teamBreakdown: Record<string, number>;
  distinctTeamsCount: number;
  bigTeamsPlayersCount: number;
  formationFeasibility: Record<string, boolean>;
  diagnostics: Record<string, unknown>;
}

export interface MultiStrategyResult {
  results: Record<string, OptimizationResult>;
}

export interface StrategyProfile {
  name: string;
  roleWeight: Record<string, number>;
  minBudgetShareByRoles: [string[], number] | null;
  maxTopTierPlayers: number | null;
  topTierCostThreshold: number | null;
}

export interface DefaultStrategiesResponse {
  strategies: StrategyProfile[];
}

// ── API Error (RFC 7807 Problem Details) ───────────────────
export interface ProblemDetails {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance?: string;
}
