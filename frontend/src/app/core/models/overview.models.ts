/** Unified player overview — Mantra ML + Hybrid ML + Gruppo Esperti + titolarità
 *  (real scraped, ML, and expert), one row per player. Served by
 *  GET /overview/players (api/src/routers/overview.py), which merges the
 *  same sources as /mantra/players and /predictions/hybrid server-side. */

export interface OverviewPlayer {
  fantacalcioId: number;
  playerFotmobId: number | null;
  seasonStart: number | null;
  playerName: string | null;
  team: string | null;
  ruoloPrimario: string | null;
  ruoliMantra: string[] | null;

  // MANTRA pillars
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
  /** Quotazione ufficiale corrente da listone — stesso valore mostrato in Players. */
  Pz1: number | null;
  /** Prezzo massimo consigliato calcolato da MANTRA — non mostrato in tabella, solo nel drawer. */
  prezzoMassimo: number | null;
  /** Asse Rendimento/Affidabilità: TOP | CERTEZZA | SCOMMESSA | null. */
  Fase7_Rendimento: string | null;
  Fase7_Rendimento_Motivo?: string | null;
  Fase7_Rendimento_Gap?: number | null;
  /** Asse Prezzo/Valore: AFFARE | GIUSTO | SOPRAVALUTATO | null. */
  Fase7_Prezzo: string | null;
  Fase7_Prezzo_Motivo?: string | null;
  Fase7_Prezzo_Gap?: number | null;
  rischio: string | null;

  // Hybrid ML
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
  /** Prediction uses career/foreign-league fallback stats rather than Serie A history. */
  isForeignFallback?: boolean;
  /** PR9: True when sample_cohort != STANDARD — prediction based on limited minutes. */
  mlValuesNoisy?: boolean;
  sampleCohort?: string | null;
  /** Shrinkage-damped predicted fantavoto for UI when sample is limited. */
  predictedDisplay?: number | null;

  // Titolarità — 3 distinct signals, never merged into one
  statusScraped: string | null;
  probabilityScraped: number | null;
  startProbability: number | null;

  // Gruppo Esperti (most recent rating for the season)
  expertRating: number | null;
  expertName: string | null;
  expertComment: string | null;
  expertTitolarita: number | null;
  expertMediaVoto: number | null;
  expertSalute: number | null;
  expertBonusLabel: string | null;
  expertBonusValue: number | null;
  expertTotale: number | null;
  expertUrl: string | null;
  expertMatchday: number | null;
}

/** One combined sort criterion. Priority is purely positional — the array
 *  index in `OverviewComponent.sortKeys`, no explicit priority field. */
export interface SortKey {
  column: string;
  direction: 'asc' | 'desc';
}

export interface OverviewPlayersResponse {
  total: number;
  page: number;
  size: number;
  items: OverviewPlayer[];
  meta?: {
    season_start: number;
    generated_at: string;
    n_players: number;
  };
}
