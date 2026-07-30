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
  varBlend?: number;
  esvWeight?: number;
  valuationMode?: 'PER_MATCH_RATING' | 'SEASON_VALUE';
  minStartProbability?: number | null;
  replacementMethod?: 'percentile' | 'roster_depth';
  strategyNames?: string[] | null;
  customStrategies?: StrategyProfile[] | null;
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
  winProbability: number | null;
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

// ── Hybrid MANTRA+ML Predictions ───────────────────────────
export interface HybridPlayerPrediction {
  playerName: string | null;
  team: string | null;
  ruoloPrimario: string | null;
  ruoliMantra: string[] | null;
  P1: number | null;
  P2: number | null;
  P3: number | null;
  P4: number | null;
  CP: number | null;
  FP: number | null;
  FP_Corr: number | null;
  CP_Corr: number | null;
  FP_Mantra: number | null;
  VR: number | null;
  Prezzo_Massimo: number | null;
  Fase7: string | null;
  rischio: string | null;
  hasMlData: boolean;
  predictedFantavoto: number | null;
  predictionStd: number | null;
  expectedMinutes: number | null;
  varScore: number | null;
  esv: number | null;
  nextSeasonPredicted: number | null;
  fpIbrido: number | null;
  mlScoreNorm: number | null;
  confidenceScore: number | null;
  mlBoost: number | null;
  fpGap: number | null;
  expectedValue: number | null;
  hybridLabels: string[] | null;
}

export interface HybridPredictionsResponse {
  total: number;
  page: number;
  size: number;
  items: HybridPlayerPrediction[];
  meta: {
    seasonStart: number;
    generatedAt: string;
    config: Record<string,number>;
    nPlayersWithMl: number;
    nPlayersWithoutMl: number;
  };
}

export interface HybridStatsResponse {
  totalPlayers: number;
  pctWithMl: number;
  avgFpIbrido: number;
  avgConfidenceScore: number;
  avgFpGap: number;
  classificationCounts: Record<string,number>;
}

export interface HybridStatus {
  mlPredictionsReady: boolean;
  mantraResults: { season: number; path: string }[];
  hybridResults: { season: number; path: string }[];
  hybridReady: boolean;
}

export interface HybridConfig {
  PESO_MANTRA: number;
  PESO_ML: number;
  W_PREDICTION_STD: number;
  W_MINUTES: number;
  EV_SCALE_FACTOR: number;
  CONFIDENZA_SOGLIA: number;
  ML_BOOST_SOGLIA: number;
  ML_BOOST_FP_CORR_MAX: number;
  ML_TOP_PRED_MIN: number;
  ML_TOP_BOOST_MIN: number;
  SOGLIA_GAP_ALERT: number;
  SLEEPER_FP_CORR_MAX: number;
  SLEEPER_ML_NORM_MIN: number;
  BEST_VALUE_VR_MIN: number;
  BEST_VALUE_FP_IBRIDO_MIN: number;
  MINUTES_RISK_MAX: number;
}
