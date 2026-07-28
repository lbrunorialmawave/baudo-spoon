import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  FotmobSearchResponse,
  IdMappingListResponse,
  IdMappingRunResponse,
  IdMappingStatsResponse,
  ManualResolutionListResponse,
  ManualResolutionStatsResponse,
  PlayerIdMapping,
  UpdateIdMappingRequest,
} from '../models/quotations.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class IdMappingService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);
  private readonly endpoint = `${this.baseUrl}/intelligence/id-mapping`;

  /** Elenco paginato dei mapping (richiede API key). */
  list(opts: {
    seasonStart?: number;
    matchMethod?: string;
    canonicalRole?: string;
    mantraRole?: string;
    matchedOnly?: boolean;
    unresolvedOnly?: boolean;
    page?: number;
    size?: number;
  } = {}): Observable<IdMappingListResponse> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.seasonStart != null) params = params.set('season_start', opts.seasonStart);
    if (opts.matchMethod)         params = params.set('match_method', opts.matchMethod);
    if (opts.canonicalRole)       params = params.set('canonical_role', opts.canonicalRole);
    if (opts.mantraRole)          params = params.set('mantra_role', opts.mantraRole);
    if (opts.matchedOnly)         params = params.set('matched_only', 'true');
    if (opts.unresolvedOnly)      params = params.set('unresolved_only', 'true');
    return this.http.get<IdMappingListResponse>(this.endpoint, { params });
  }

  /** Statistiche aggregate dei mapping. */
  getStats(): Observable<IdMappingStatsResponse> {
    return this.http.get<IdMappingStatsResponse>(`${this.endpoint}/stats`);
  }

  /** Singolo mapping per fantacalcio_id (+ optional season_start). */
  get(fantacalcioId: number, seasonStart?: number): Observable<PlayerIdMapping> {
    let params = new HttpParams();
    if (seasonStart != null) params = params.set('season_start', seasonStart);
    return this.http.get<PlayerIdMapping>(`${this.endpoint}/${fantacalcioId}`, { params });
  }

  /** Aggiornamento manuale di un mapping. */
  update(
    fantacalcioId: number,
    seasonStart: number,
    body: UpdateIdMappingRequest,
  ): Observable<PlayerIdMapping> {
    const params = new HttpParams().set('season_start', seasonStart);
    return this.http.put<PlayerIdMapping>(`${this.endpoint}/${fantacalcioId}`, body, { params });
  }

  // ── Manual Resolution History ──────────────────────────────────────────

  /** Elenco paginato delle risoluzioni manuali storiche. */
  listResolutions(opts: {
    seasonStart?: number;
    fantacalcioId?: number;
    playerFotmobId?: number;
    search?: string;
    page?: number;
    size?: number;
  } = {}): Observable<ManualResolutionListResponse> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.seasonStart != null)    params = params.set('season_start', opts.seasonStart);
    if (opts.fantacalcioId != null)  params = params.set('fantacalcio_id', opts.fantacalcioId);
    if (opts.playerFotmobId != null) params = params.set('player_fotmob_id', opts.playerFotmobId);
    if (opts.search)                 params = params.set('search', opts.search);
    return this.http.get<ManualResolutionListResponse>(`${this.endpoint}/resolutions`, { params });
  }

  /** Statistiche aggregate delle risoluzioni manuali. */
  getResolutionStats(): Observable<ManualResolutionStatsResponse> {
    return this.http.get<ManualResolutionStatsResponse>(`${this.endpoint}/resolutions/stats`);
  }

  /** Elimina una risoluzione manuale. */
  deleteResolution(id: number): Observable<{ status: string; id: number }> {
    return this.http.delete<{ status: string; id: number }>(`${this.endpoint}/resolutions/${id}`);
  }

  /** Esporta tutti i mapping come array JSON (per il merger ibrido). */
  exportMappings(): Observable<Array<{
    fantacalcio_id: number;
    player_fotmob_id: number;
    season_start: number;
    match_method: string;
  }>> {
    return this.http.get<Array<{
      fantacalcio_id: number;
      player_fotmob_id: number;
      season_start: number;
      match_method: string;
    }>>(`${this.endpoint}/export`);
  }

  /** Avvia il pipeline automatico di ID mapping. */
  runAutoMapping(opts: {
    seasonStart?: number;
    leagueName?: string;
  } = {}): Observable<IdMappingRunResponse> {
    let params = new HttpParams();
    if (opts.seasonStart != null) params = params.set('season_start', opts.seasonStart);
    if (opts.leagueName)          params = params.set('league_name', opts.leagueName);
    return this.http.post<IdMappingRunResponse>(`${this.endpoint}/run`, null, { params });
  }

  /** Cerca giocatori su FotMob tramite suggest API. */
  fotmobSearch(term: string, hits: number = 10): Observable<FotmobSearchResponse> {
    const params = new HttpParams()
      .set('term', term)
      .set('hits', hits);
    return this.http.get<FotmobSearchResponse>(`${this.endpoint}/fotmob-search`, { params });
  }
}
