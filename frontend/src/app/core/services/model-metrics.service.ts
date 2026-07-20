import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';
import {
  CompareResponse,
  MetricPoint,
  ModelRunsResponse,
} from '../models/model-metrics.models';

@Injectable({ providedIn: 'root' })
export class ModelMetricsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getRuns(modelName?: string, limit = 20, offset = 0): Observable<ModelRunsResponse> {
    let params = new HttpParams().set('limit', limit).set('offset', offset);
    if (modelName) params = params.set('model_name', modelName);
    return this.http.get<ModelRunsResponse>(`${this.baseUrl}/model-metrics/runs`, { params });
  }

  getHistory(metric = 'rmse', split = 'test', modelName?: string): Observable<MetricPoint[]> {
    let params = new HttpParams().set('metric', metric).set('split', split);
    if (modelName) params = params.set('model_name', modelName);
    return this.http.get<MetricPoint[]>(`${this.baseUrl}/model-metrics/history`, { params });
  }

  compare(runA: string, runB: string): Observable<CompareResponse> {
    const params = new HttpParams().set('run_a', runA).set('run_b', runB);
    return this.http.get<CompareResponse>(`${this.baseUrl}/model-metrics/compare`, { params });
  }
}
