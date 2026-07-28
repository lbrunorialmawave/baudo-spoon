// NOTE: All fields are camelCase.
// Quotation endpoints use _CamelModel + model_dump(by_alias=True).

export interface PlayerQuotation {
  id: number;
  fantacalcioId: number;
  seasonStart: number;
  role: 'GK' | 'DEF' | 'MID' | 'FWD';
  team: string;
  playerName: string;
  qtA: number;
  qtI: number;
  diffVal: number;
  qtAM: number | null;
  qtIM: number | null;
  diffValM: number | null;
  fvm: number | null;
  fvmM: number | null;
  source: string;
  importedAt: string;
  playerFotmobId: number | null;
  nameFotmob: string | null;
  teamFotmob: string | null;
  matchMethod: string | null;
  confidence: number | null;
  ruoloPrimario: string | null;
  ruoliMantra: string[] | null;
}

export interface QuotationRoleAggregate {
  seasonStart: number;
  role: string;
  nPlayers: number;
  avgQtA: number;
  avgQtI: number;
  medianQtA: number;
  minQtA: number;
  maxQtA: number;
  avgFvm: number | null;
}

export interface QuotationStatsResponse {
  totalQuotations: number;
  seasons: number[];
  bySeasonRole: QuotationRoleAggregate[];
  nTeams: number;
  coverage: Record<string, number>;
}

export interface QuotationListResponse {
  total: number;
  page: number;
  size: number;
  items: PlayerQuotation[];
}

export interface QuotationSeasonResponse {
  seasonStart: number;
  total: number;
  items: PlayerQuotation[];
}

export interface QuotationPlayerHistoryResponse {
  playerFotmobId: number;
  total: number;
  items: PlayerQuotation[];
}

// ── ID Mapping (Fantacalcio ↔ FotMob) ────────────────────────────────────────

export interface PlayerIdMapping {
  id: number;
  fantacalcioId: number;
  seasonStart: number;
  playerFotmobId: number | null;
  nameFantacalcio: string;
  nameFotmob: string | null;
  teamFantacalcio: string | null;
  teamFotmob: string | null;
  canonicalRole: string | null;
  matchMethod: string;
  confidence: number;
  resolvedFromHistory?: boolean;
  createdAt: string;
  updatedAt: string;
  // MANTRA 12-role fields
  ruoliMantra?: string[] | null;
  ruoloPrimario?: string | null;
}

export interface IdMappingListResponse {
  total: number;
  page: number;
  size: number;
  items: PlayerIdMapping[];
}

export interface IdMappingStatsResponse {
  total: number;
  matched: number;
  unmatched: number;
  matchRate: number;
  bySeason: Record<string, Record<string, number>>;
  byMethod: Record<string, number>;
}

// ── Manual Resolution History ──────────────────────────────────────────────

export interface ManualResolution {
  id: number;
  fantacalcioId: number;
  playerFotmobId: number;
  seasonStart: number;
  nameFantacalcio: string;
  teamFantacalcio: string | null;
  canonicalRole: string | null;
  nameFotmob: string | null;
  teamFotmob: string | null;
  resolvedBy: string | null;
  note: string | null;
  createdAt: string;
}

export interface ManualResolutionListResponse {
  total: number;
  page: number;
  size: number;
  items: ManualResolution[];
}

export interface ManualResolutionStatsResponse {
  total: number;
  uniquePlayers: number;
  bySeason: Record<string, number>;
}

/** Request body per aggiornare manualmente un mapping. */
export interface UpdateIdMappingRequest {
  playerFotmobId?: number | null;
  nameFotmob?: string | null;
  teamFotmob?: string | null;
  canonicalRole?: string | null;
  note?: string | null;
  // MANTRA role overrides (optional)
  ruoliMantra?: string[] | null;
  ruoloPrimario?: string | null;
  dataValidated?: boolean | null;
}

/** Risposta dal pipeline automatico di ID mapping. */
export interface IdMappingRunResponse {
  status: string;
  total: number;
  matched: number;
  unmatched: number;
  matchRatePct: number;
  byMethod: Record<string, number>;
}

// ── FotMob Search (suggest API) ─────────────────────────────────────────────

export interface FotmobSearchItem {
  playerFotmobId: number;
  name: string;
  teamId: number | null;
  teamName: string | null;
  score: number;
}

export interface FotmobSearchResponse {
  term: string;
  total: number;
  items: FotmobSearchItem[];
}
