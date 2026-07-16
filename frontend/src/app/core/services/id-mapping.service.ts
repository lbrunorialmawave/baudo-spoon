import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  IdMappingListResponse,
  IdMappingStatsResponse,
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
    matchedOnly?: boolean;
    page?: number;
    size?: number;
  } = {}): Observable<IdMappingListResponse> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.seasonStart != null) params = params.set('season_start', opts.seasonStart);
    if (opts.matchMethod)         params = params.set('match_method', opts.matchMethod);
    if (opts.canonicalRole)       params = params.set('canonical_role', opts.canonicalRole);
    if (opts.matchedOnly)         params = params.set('matched_only', 'true');
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
}
