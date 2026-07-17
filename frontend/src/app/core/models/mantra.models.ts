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
  Fase7: string | null;
  rischio: string | null;
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
  total_rows: number;
  status: string;
  match_rate_pct?: number;
  latest_matchday?: number;
  seasons?: number[];
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

export const MANTRA_ROLES = [
  'Por', 'Dc', 'Dd', 'Ds', 'B', 'E', 'M', 'C', 'T', 'W', 'A', 'Pc',
];

export const MATCHDAY_STATUS_CONFIG: Record<string, { label: string; color: string }> = {
  starter:   { label: 'Titolare',    color: '#22C55E' },
  bench:     { label: 'Panchina',    color: '#6B7280' },
  injured:   { label: 'Infortunato', color: '#EF4444' },
  suspended: { label: 'Squalificato',color: '#EF4444' },
  doubtful:  { label: 'In dubbio',   color: '#F59E0B' },
  unknown:   { label: 'Sconosciuto', color: '#9CA3AF' },
};
