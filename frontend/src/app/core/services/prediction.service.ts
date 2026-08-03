import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  NextSeasonPrediction,
  PredictionsResponse,
  HybridPredictionsResponse,
  HybridStatsResponse,
  HybridConfig,
  HybridStatus,
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

  getHybridStatus(): Observable<HybridStatus> {
    return this.http.get<HybridStatus>(`${this.baseUrl}/predictions/hybrid/status`);
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

  // ── ML Pipeline runs ─────────────────────────────────

  getPipelineRuns(limit?: number, offset?: number): Observable<{
    items: Array<{
      run_id: string;
      model_name: string;
      trained_at: string;
      season_start: number;
      git_commit: string | null;
      status: string;
      metrics: Array<{ metric: string; value: number; split: string }>;
    }>;
    offset: number;
    limit: number;
  }> {
    let params = new HttpParams();
    if (limit) params = params.set('limit', limit);
    if (offset) params = params.set('offset', offset);
    return this.http.get<any>(`${this.baseUrl}/model-metrics/runs`, { params });
  }

  /** Trigger the "ML Training" GitHub Actions workflow (admin only). Runs on
   *  GitHub's own runner, not locally — see api/src/routers/ml_pipeline.py. */
  trainModel(): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.baseUrl}/admin/ml/train`, {});
  }

  /** Poll while a training run is in progress. Reflects the most recent
   *  GitHub Actions run of the ML Training workflow. */
  getTrainingStatus(): Observable<{
    status: 'idle' | 'running' | 'completed' | 'failed';
    run_number?: number;
    started_at?: string;
    updated_at?: string;
    conclusion?: string | null;
    html_url?: string;
  }> {
    return this.http.get<any>(`${this.baseUrl}/admin/ml/train/status`);
  }

  invalidateCache(): Observable<{ detail: string }> {
    return this.http.post<{ detail: string }>(`${this.baseUrl}/intelligence/cache/invalidate`, {});
  }
}
