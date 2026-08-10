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
  Fase7: string | null;
  /** Only populated when Fase7 is null — explains why no rule matched. */
  Fase7_Motivo?: string | null;
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
  fase7_distribution: Record<string, number>;
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
  fase7?: string;
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
  /** Present on neo_arrivi_coverage source (P5). */
  season_start?: number;
  unmatched_total?: number;
  resolved_by_retry?: number;
  foreign_stats_candidates?: number;
  reason?: string;
}

export interface DataHealthResponse {
  sources: DataHealthSource[];
}

export const FASE7_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  TOP:            { label: 'TOP',            color: '#F59E0B', icon: '🏆' },
  AFFARE:         { label: 'AFFARE',         color: '#22C55E', icon: '💎' },
  SCOMMESSA:      { label: 'SCOMMESSA',      color: '#8B5CF6', icon: '🔄' },
  CERTEZZA:       { label: 'CERTEZZA',       color: '#06B6D4', icon: '✅' },
  SOPRAVALUTATO:  { label: 'SOPRAVALUTATO',  color: '#EF4444', icon: '⚠️' },
  GIUSTO:         { label: 'GIUSTO',         color: '#6B7280', icon: '⚖️' },
};

/** Single source of truth for the 6 "Profilo" (Fase 7) category explanations —
 *  used by the legend, the quick-filter buttons, and the row badges. */
export const FASE7_TOOLTIPS: Record<string, string> = {
  TOP:            '🏆 TOP — Giocatore d\'élite: FP alto e VR bilanciato. Investimento sicuro.',
  AFFARE:         '💎 AFFARE — Sottovalutato dal mercato: FP alto, prezzo basso. Ottimo rapporto Q/P.',
  SCOMMESSA:      '🔄 SCOMMESSA — Potenziale inespresso: FP basso ma VR alto. Può esplodere.',
  CERTEZZA:       '✅ CERTEZZA — Rendimento stabile e affidabile. Poche sorprese.',
  SOPRAVALUTATO:  '⚠️ SOPRAVALUTATO — Prezzo gonfiato: VR basso rispetto al FP. Rischi.',
  GIUSTO:         '⚖️ GIUSTO — Nella media: FP e VR allineati al prezzo di mercato.',
};

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
