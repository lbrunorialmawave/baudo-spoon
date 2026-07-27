import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  NextSeasonPrediction,
  PredictionsResponse,
  HybridPredictionsResponse,
  HybridStatsResponse,
  HybridConfig,
} from '../models/api.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getPredictions(): Observable<PredictionsResponse> {
    const params = new HttpParams().set('page', '1').set('size', '200');
    return this.http.get<PredictionsResponse>(`${this.baseUrl}/predictions/players`, { params });
  }

  getNextSeason(player?: string): Observable<NextSeasonPrediction[]> {
    let params = new HttpParams();
    if (player) params = params.set('player', player);
    return this.http.get<NextSeasonPrediction[]>(`${this.baseUrl}/predictions/next-season`, { params });
  }

  // ── Hybrid MANTRA+ML ────────────────────────────────────

  getHybridPredictions(params?: {
    page?: number;
    size?: number;
    ruolo?: string;
    search?: string;
    confidenceMin?: number;
    label?: string;
    sortBy?: string;
    sortDir?: string;
  }): Observable<HybridPredictionsResponse> {
    let httpParams = new HttpParams();
    if (params) {
      if (params.page) httpParams = httpParams.set('page', params.page);
      if (params.size) httpParams = httpParams.set('size', params.size);
      if (params.ruolo) httpParams = httpParams.set('ruolo', params.ruolo);
      if (params.search) httpParams = httpParams.set('search', params.search);
      if (params.confidenceMin !== undefined) httpParams = httpParams.set('confidenceMin', params.confidenceMin);
      if (params.label) httpParams = httpParams.set('label', params.label);
      if (params.sortBy) httpParams = httpParams.set('sortBy', params.sortBy);
      if (params.sortDir) httpParams = httpParams.set('sortDir', params.sortDir);
    }
    return this.http.get<HybridPredictionsResponse>(`${this.baseUrl}/predictions/hybrid`, { params: httpParams });
  }

  getHybridStats(): Observable<HybridStatsResponse> {
    return this.http.get<HybridStatsResponse>(`${this.baseUrl}/predictions/hybrid/stats`);
  }

  getHybridConfig(): Observable<HybridConfig> {
    return this.http.get<HybridConfig>(`${this.baseUrl}/predictions/hybrid/config`);
  }

  updateHybridConfig(config: Partial<HybridConfig>): Observable<HybridConfig> {
    return this.http.put<HybridConfig>(`${this.baseUrl}/predictions/hybrid/config`, config);
  }

  runHybrid(seasonStart: number = 2025, overrides?: Partial<HybridConfig>, persist: boolean = true): Observable<{
    status: string;
    season: number;
    nPlayers: number;
    generatedAt: string;
    persisted: boolean;
  }> {
    let params = new HttpParams()
      .set('season_start', seasonStart)
      .set('persist', persist);
    return this.http.post<any>(`${this.baseUrl}/predictions/hybrid/run`, overrides || {}, { params });
  }

  getHybridPreview(seasonStart?: number): Observable<HybridPredictionsResponse> {
    let params = new HttpParams();
    if (seasonStart) params = params.set('seasonStart', seasonStart);
    return this.http.get<HybridPredictionsResponse>(`${this.baseUrl}/predictions/hybrid/preview`, { params });
  }
}
