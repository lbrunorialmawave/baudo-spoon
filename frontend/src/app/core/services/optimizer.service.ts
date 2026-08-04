import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';
import {
  DefaultStrategiesResponse,
  MultiStrategyResult,
  OptimizationRequest,
  OptimizationResult,
  OptimizeJobCreateResponse,
  OptimizeJobStatus,
} from '../models/api.models';

/**
 * Optimizer API client.
 *
 * Contract notes (FAANG-style):
 * - Deterministic path is the default (omit monteCarlo / enabled=false).
 * - SAA with large N should prefer createJob + pollJobStatus over runMulti.
 * - All request/response field names are camelCase (backend _CamelModel).
 */
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

  runSingle(
    req: OptimizationRequest,
    strategyName = 'BALANCED',
  ): Observable<OptimizationResult> {
    return this.http.post<OptimizationResult>(
      `${this.baseUrl}/optimize/single`,
      req,
      { params: { strategy_name: strategyName } },
    );
  }

  /** Enqueue async MC job. Requires monteCarlo.enabled=true. */
  createJob(
    req: OptimizationRequest,
    strategyName = 'BALANCED',
  ): Observable<OptimizeJobCreateResponse> {
    return this.http.post<OptimizeJobCreateResponse>(
      `${this.baseUrl}/optimize/jobs`,
      req,
      { params: { strategy_name: strategyName } },
    );
  }

  pollJobStatus(jobId: string): Observable<OptimizeJobStatus> {
    return this.http.get<OptimizeJobStatus>(`${this.baseUrl}/optimize/jobs/${jobId}`);
  }
}
