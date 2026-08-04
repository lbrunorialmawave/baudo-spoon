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
  /**
   * Peso dell'aggiustamento Elo di Club sulla stima del costo di un
   * giocatore. 0 = disattivato (default backend), valori più alti
   * premiano i giocatori di squadre con Elo alto nel prezzo stimato
   * (costa di più per le big, costa di meno per le piccole).
   * Range suggerito: [0, 1.5]. Valori negativi non hanno effetto
   * (Pydantic accetta solo `ge=0.0`).
   */
  teamStrengthMultiplier?: number;
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
  /** Monte Carlo robustness block. Omit or enabled=false keeps deterministic ILP. */
  monteCarlo?: MonteCarloConfig | null;
  /** Near-optimal alternatives (exclude top scorers, re-solve). Default off. */
  nearOptimal?: NearOptimalConfig | null;
}

/** Request block for score-space Monte Carlo robustness. */
export interface MonteCarloConfig {
  enabled: boolean;
  /** SAA scenarios; sync path typically 5–50, hard-capped by API. */
  nSimulations?: number;
  /** mean_std = one risk-adjusted ILP; saa_frequency = N scenario ILPs + frequency. */
  mode?: 'mean_std' | 'saa_frequency';
  /** Only for mean_std: mean − λ·std. */
  riskLambda?: number;
  minSelectionFrequency?: number;
  randomSeed?: number;
  /** Soft wall budget for SAA; 0 = server default. */
  timeoutSeconds?: number;
}

export interface NearOptimalConfig {
  enabled: boolean;
  nAlternatives?: number;
  excludeTopM?: number;
  maxScoreDropPct?: number;
}

export interface YieldStabilitySummary {
  nSimulations?: number;
  threshold?: number;
  probAboveThreshold?: number;
  meanTotal?: number;
  p10Total?: number;
  p50Total?: number;
  p90Total?: number;
}

export interface MonteCarloSummary {
  nSimulations: number;
  mode: string;
  randomSeed?: number;
  stabilityIndex?: number;
  selectionFrequency?: Record<string, number>;
  squadScorePercentiles?: Record<string, number>;
  meanPairwiseJaccard?: number;
  scenariosCompleted?: number;
  wallTimeSeconds?: number;
  samplingMethodsCounts?: Record<string, number>;
  warnings?: string[];
  yieldStability?: YieldStabilitySummary | null;
}

export interface NearOptimalAlternative {
  excludedPlayerIds: string[];
  scoreDelta: number;
  scoreDeltaPct: number;
  squad: SquadPlayer[];
  totalProjectedScore: number;
  status: string;
}

export interface DiversityMetrics {
  meanPairwiseJaccard: number;
  maxPairwiseJaccard?: number;
  minPairwiseJaccard?: number;
  meanOverlapCount?: number;
  maxOverlapCount?: number;
  lowDiversity: boolean;
  pairwiseJaccard?: Record<string, number>;
}

export interface OptimizeJobCreateResponse {
  jobId: string;
  status: string;
}

export interface OptimizeJobStatus {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | string;
  createdAt: string;
  updatedAt: string;
  error?: string | null;
  result?: OptimizationResult | null;
  monteCarloSummary?: MonteCarloSummary | null;
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
  monteCarloSummary?: MonteCarloSummary | null;
  nearOptimal?: NearOptimalAlternative[];
}

export interface MultiStrategyResult {
  results: Record<string, OptimizationResult>;
  monteCarloSummary?: MonteCarloSummary | null;
  diversity?: DiversityMetrics | null;
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
