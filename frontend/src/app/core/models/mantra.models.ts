/** MANTRA scoring models — player evaluations with 4 pillars. */

export interface MantraPlayer {
  fantacalcio_id: number;
  season_start: number;
  player_name: string;
  team: string;
  ruolo_primario: string;
  ruoli_mantra: string[];
  Pz1: number;  // qt_a — newspaper auction price
  Pz2: number;  // qt_i — initial quotation
  Pz3: number;  // fvm  — fantacalcio market value
  P1: number;
  P2: number;
  P3: number;
  P4: number;
  CP: number;
  FP: number;
  FP_Corr: number;
  CP_Corr: number;
  FP_Mantra: number;
  VR: number;
  Prezzo_Massimo: number;
  /** Percentile (0-1) di FP_Mantra nel pool esteso del ruolo del giocatore. */
  Percentile_Ruolo?: number;
  /** Asse Rendimento/Affidabilità: TOP | CERTEZZA | SCOMMESSA | null. */
  Fase7_Rendimento: string | null;
  /** Only populated when Fase7_Rendimento is null — explains why no rule matched. */
  Fase7_Rendimento_Motivo?: string | null;
  /** Percentile(VR) - percentile(FP) nel pool di ruolo — confidenza numerica
   *  per l'asse Rendimento (usata per le stelle su SCOMMESSA). */
  Fase7_Rendimento_Gap?: number | null;
  /** Asse Prezzo/Valore: AFFARE | GIUSTO | SOPRAVALUTATO | null. */
  Fase7_Prezzo: string | null;
  /** Only populated when Fase7_Prezzo is null — explains why no rule matched. */
  Fase7_Prezzo_Motivo?: string | null;
  /** Percentile(prezzo) - percentile(VR) nel pool di ruolo — confidenza
   *  numerica per l'asse Prezzo (usata per le stelle su AFFARE/SOPRAVALUTATO). */
  Fase7_Prezzo_Gap?: number | null;
  rischio: string | null;
  season_value?: number | null;
  start_probability?: number | null;
}

export interface MantraPlayersResponse {
  total: number;
  page: number;
  size: number;
  items: MantraPlayer[];
  meta?: {
    season_start: number;
    generated_at: string;
    n_players: number;
  };
}

export interface MantraStatsResponse {
  total_players: number;
  season_start: number;
  avg_fp_mantra: number;
  avg_vr: number;
  fase7_rendimento_distribution: Record<string, number>;
  fase7_prezzo_distribution: Record<string, number>;
}

export interface MantraTopResponse {
  ruolo: string;
  limit: number;
  items: MantraPlayer[];
}

export interface MatchdayPlayerStatus {
  fantacalcio_id: number;
  season_start: number;
  matchday: number;
  team: string;
  probability: number;
  status: string;
  injury_note: string | null;
  player_name: string | null;
  ruolo_primario: string | null;
  ruoli_mantra: string[] | null;
  fp_mantra?: number;
  vr?: number;
  fase7_rendimento?: string;
  fase7_prezzo?: string;
}

export interface DataHealthSource {
  name: string;
  total_rows?: number;
  status: string;
  match_rate_pct?: number;
  matched?: number;
  unmatched?: number;
  latest_matchday?: number;
  seasons?: number[];
  /** Present on neo_arrivi_coverage / ml_coverage sources. */
  season_start?: number;
  unmatched_total?: number;
  resolved_by_retry?: number;
  foreign_stats_candidates?: number;
  reason?: string;
  /** Present on ml_coverage source (GET /admin/data-health). */
  coverage_pct?: number;
  n_players?: number;
  n_with_ml_data?: number;
  n_neo_arrivo?: number;
  n_neo_arrivo_unresolved?: number;
  /** "results_latest.json" when present, otherwise "missing". */
  artifact?: string;
  warning_threshold_pct?: number;
  neo_arrivo_unresolved?: Array<{
    fantacalcio_id: number;
    player_name?: string;
    team?: string;
    player_fotmob_id?: number | null;
  }>;
}

export interface DataHealthResponse {
  sources: DataHealthSource[];
}

/** Fase7 Asse Rendimento/Affidabilità: TOP > CERTEZZA > SCOMMESSA (mutually
 *  exclusive within this axis only — see FASE7_PREZZO_LABELS for the other,
 *  independent axis). */
export const FASE7_RENDIMENTO_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  TOP:            { label: 'TOP',            color: '#F59E0B', icon: '🏆' },
  CERTEZZA:       { label: 'CERTEZZA',       color: '#06B6D4', icon: '✅' },
  SCOMMESSA:      { label: 'SCOMMESSA',      color: '#8B5CF6', icon: '🔄' },
};

/** Fase7 Asse Prezzo/Valore: tre fasce contigue di un unico gap tra
 *  quotazione e VR — indipendente dall'asse Rendimento/Affidabilità. */
export const FASE7_PREZZO_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  AFFARE:         { label: 'AFFARE',         color: '#22C55E', icon: '💎' },
  GIUSTO:         { label: 'GIUSTO',         color: '#6B7280', icon: '⚖️' },
  SOPRAVALUTATO:  { label: 'SOPRAVALUTATO',  color: '#EF4444', icon: '⚠️' },
};

/** Combined lookup for UI surfaces (e.g. a single filter dropdown) that
 *  don't need to distinguish the two axes — each key is still unique
 *  across both maps. */
export const FASE7_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  ...FASE7_RENDIMENTO_LABELS,
  ...FASE7_PREZZO_LABELS,
};

/** Which axis a Fase7 label key belongs to — lets a combined dropdown/quick
 *  filter route the selected value to the right query param
 *  (fase7Rendimento vs fase7Prezzo). */
export const FASE7_AXIS: Record<string, 'rendimento' | 'prezzo'> = {
  TOP: 'rendimento',
  CERTEZZA: 'rendimento',
  SCOMMESSA: 'rendimento',
  AFFARE: 'prezzo',
  GIUSTO: 'prezzo',
  SOPRAVALUTATO: 'prezzo',
};

/** Single source of truth for the 6 "Profilo" (Fase 7) category explanations —
 *  used by the legend, the quick-filter buttons, and the row badges. */
export const FASE7_TOOLTIPS: Record<string, string> = {
  TOP:            '🏆 TOP — Giocatore d\'élite: FP alto e VR bilanciato. Investimento sicuro.',
  CERTEZZA:       '✅ CERTEZZA — Rendimento stabile e affidabile (storico, o titolarità attesa blindata). Poche sorprese.',
  SCOMMESSA:      '🔄 SCOMMESSA — Potenziale inespresso: FP basso ma VR nettamente più alto. Può esplodere.',
  AFFARE:         '💎 AFFARE — Prezzo nettamente sotto il VR nel pool di ruolo. Ottimo rapporto Q/P.',
  GIUSTO:         '⚖️ GIUSTO — Prezzo allineato al VR nel pool di ruolo.',
  SOPRAVALUTATO:  '⚠️ SOPRAVALUTATO — Prezzo nettamente sopra il VR nel pool di ruolo. Rischi.',
};

/** Derive a 1-3 "star" confidence rating from a Fase7 gap value (percentile
 *  points). `baseThreshold` is the gap magnitude at which the label itself
 *  first kicks in (SCOMMESSA_GAP_MIN=25 for the Rendimento axis,
 *  GIUSTO_GAP_BAND=15 for the Prezzo axis in ml/mantra/config.py — keep
 *  these two numbers in sync with the backend if they're ever retuned).
 *  Returns null when there's no gap to rate (e.g. never-quoted player). */
export function fase7Stars(gap: number | null | undefined, baseThreshold: number): 1 | 2 | 3 | null {
  if (gap == null || !Number.isFinite(gap)) return null;
  const magnitude = Math.abs(gap);
  if (magnitude < baseThreshold) return null;
  if (magnitude < baseThreshold * 2) return 1;
  if (magnitude < baseThreshold * 3) return 2;
  return 3;
}

export const MANTRA_ROLES = [
  'Por', 'Dc', 'Dd', 'Ds', 'B', 'E', 'M', 'C', 'T', 'W', 'A', 'Pc',
];

/** Hybrid MANTRA+ML classification labels — shared by the Predictions
 *  "Ibrido" tab and the Overview page (both filter/render the same
 *  `hybridLabels` field off HybridPlayerPrediction / OverviewPlayer). */
export const HYBRID_LABELS: { id: string; label: string; color: string; desc: string }[] = [
  { id: 'ML_Confirmed', label: 'Confermato',          color: '#16a34a', desc: 'ML concorde col MANTRA, minutaggio garantito' },
  { id: 'ML_Risky',      label: 'Rischioso',           color: '#dc2626', desc: 'Prediction poco affidabile, confidence bassa' },
  { id: 'ML_Top',        label: 'Top',                 color: '#7c3aed', desc: 'Giocatore top riconosciuto dal ML' },
  { id: 'ML_Boosted',    label: 'Sorpresa',            color: '#a855f7', desc: 'ML molto sopra la media del ruolo, possibile sorpresa' },
  { id: 'Contradiction', label: 'Contrasto',           color: '#d97706', desc: 'Disaccordo MANTRA vs ML — valutare con cautela' },
  { id: 'Minutes_Risk',  label: 'Minuti a rischio',    color: '#f97316', desc: 'Pochi minuti previsti in stagione' },
  { id: 'Best_Value',    label: 'Miglior rapporto Q/P', color: '#22c55e', desc: 'Ottimo rapporto qualità/prezzo all\'asta' },
  { id: 'Sleeper',       label: 'Sleeper',             color: '#3b82f6', desc: 'Sottovalutato dal MANTRA ma con buona prediction ML' },
];

export const MATCHDAY_STATUS_CONFIG: Record<string, { label: string; color: string }> = {
  starter:   { label: 'Titolare',    color: '#22C55E' },
  bench:     { label: 'Panchina',    color: '#6B7280' },
  injured:   { label: 'Infortunato', color: '#EF4444' },
  suspended: { label: 'Squalificato',color: '#EF4444' },
  doubtful:  { label: 'In dubbio',   color: '#F59E0B' },
  unknown:   { label: 'Sconosciuto', color: '#9CA3AF' },
};
