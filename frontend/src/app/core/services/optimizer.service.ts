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
  ParetoResponse,
  SensitivityResponse,
} from '../models/api.models';

/**
 * Optimizer API client.
 *
 * Contract notes (FAANG-style):
 * - Deterministic path is the default (omit monteCarlo / enabled=false).
 * - SAA with large N should prefer createJob + pollJobStatus over runMulti.
 * - Sensitivity / Pareto are opt-in analysis endpoints: same OptimizationRequest
 *   body, but require exactly one strategy (strategyNames or customStrategies).
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

  /**
   * One-at-a-time sensitivity sweep (risk_aversion, var_blend, hybrid_blend, budget)
   * around the request baseline. Backend requires exactly one strategy entry.
   */
  runSensitivity(req: OptimizationRequest): Observable<SensitivityResponse> {
    return this.http.post<SensitivityResponse>(`${this.baseUrl}/optimize/sensitivity`, req);
  }

  /**
   * Score vs risk Pareto frontier for the baseline strategy.
   * Backend requires exactly one strategy entry.
   */
  runPareto(req: OptimizationRequest): Observable<ParetoResponse> {
    return this.http.post<ParetoResponse>(`${this.baseUrl}/optimize/pareto`, req);
  }
}
