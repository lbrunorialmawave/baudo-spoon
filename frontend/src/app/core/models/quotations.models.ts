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
