import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  QuotationListResponse,
  QuotationPlayerHistoryResponse,
  QuotationSeasonResponse,
  QuotationStatsResponse,
} from '../models/quotations.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class QuotationService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getQuotations(opts: {
    seasonStart?: number; role?: string; team?: string;
    playerFotmobId?: number; minQtA?: number; maxQtA?: number;
    ruoloPrimario?: string; ruoloMantra?: string;
    page?: number; size?: number;
  } = {}): Observable<QuotationListResponse> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.seasonStart != null)   params = params.set('season_start', opts.seasonStart);
    if (opts.role)                  params = params.set('role', opts.role);
    if (opts.team)                  params = params.set('team', opts.team);
    if (opts.playerFotmobId != null) params = params.set('player_fotmob_id', opts.playerFotmobId);
    if (opts.minQtA != null)        params = params.set('min_qt_a', opts.minQtA);
    if (opts.maxQtA != null)        params = params.set('max_qt_a', opts.maxQtA);
    if (opts.ruoloPrimario)         params = params.set('ruolo_primario', opts.ruoloPrimario);
    if (opts.ruoloMantra)           params = params.set('ruolo_mantra', opts.ruoloMantra);
    return this.http.get<QuotationListResponse>(`${this.baseUrl}/quotations`, { params });
  }

  getSeasons(): Observable<number[]> {
    return this.http.get<number[]>(`${this.baseUrl}/quotations/seasons`);
  }

  getSeasonQuotations(seasonStart: number): Observable<QuotationSeasonResponse> {
    return this.http.get<QuotationSeasonResponse>(`${this.baseUrl}/quotations/seasons/${seasonStart}`);
  }

  getPlayerHistory(playerFotmobId: number): Observable<QuotationPlayerHistoryResponse> {
    return this.http.get<QuotationPlayerHistoryResponse>(`${this.baseUrl}/quotations/players/${playerFotmobId}`);
  }

  getStats(): Observable<QuotationStatsResponse> {
    return this.http.get<QuotationStatsResponse>(`${this.baseUrl}/quotations/stats`);
  }
}
