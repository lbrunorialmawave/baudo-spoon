import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  DefaultStrategiesResponse,
  MultiStrategyResult,
  OptimizationRequest,
  OptimizationResult,
} from '../models/api.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class OptimizerService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getStrategies(): Observable<DefaultStrategiesResponse> {
    return this.http.get<DefaultStrategiesResponse>(`${this.baseUrl}/optimize/strategies`);
  }

  runMulti(req: OptimizationRequest): Observable<MultiStrategyResult> {
    return this.http.post<MultiStrategyResult>(`${this.baseUrl}/optimize/multi`, req);
  }

  runSingle(req: OptimizationRequest, strategyName: string): Observable<OptimizationResult> {
    return this.http.post<OptimizationResult>(
      `${this.baseUrl}/optimize/single`,
      req,
      { params: { strategy_name: strategyName } },
    );
  }
}
