import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';
import {
  AlternativesRequest,
  AlternativesResponse,
  AuctionPlayerSummary,
  AuctionSummary,
  DeserializeAuctionRequest,
  InitializeAuctionRequest,
  InitializeAuctionResponse,
  ProjectionResponse,
  RecordAssignmentRequest,
  RecordAssignmentResponse,
  SerializedAuctionStateResponse,
  VarRankingResponse,
} from '../models/auction.models';

/**
 * HTTP client for the live auction tracker (`/api/v1/auction`).
 *
 * The backend is *single-operator, single-process*: every method below
 * targets a server-side session identified by `sessionId`.  Multiple
 * concurrent sessions are supported (each with its own `sessionId`), but
 * a single session is not safe for parallel edits.
 */
@Injectable({ providedIn: 'root' })
export class AuctionService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  /** POST /auction/init — bootstrap a fresh auction session. */
  init(req: InitializeAuctionRequest): Observable<InitializeAuctionResponse> {
    return this.http.post<InitializeAuctionResponse>(
      `${this.baseUrl}/auction/init`,
      req,
    );
  }

  /**
   * POST /auction/{sessionId}/record — validate and register an assignment.
   *
   * The server returns HTTP 200 even on validation rejections
   * (`success=false` + `rejectionCode`); HTTP errors are reserved for
   * server-side faults.  See {@link RecordAssignmentResponse} for the
   * full set of `rejectionCode` values.
   */
  record(
    sessionId: string,
    req: RecordAssignmentRequest,
  ): Observable<RecordAssignmentResponse> {
    return this.http.post<RecordAssignmentResponse>(
      `${this.baseUrl}/auction/${sessionId}/record`,
      req,
    );
  }

  /**
   * POST /auction/{sessionId}/undo — revert the most recent assignment.
   *
   * Returns 409 (raised as `HttpErrorResponse`) when the session has no
   * assignments to undo.
   */
  undo(sessionId: string): Observable<void> {
    return this.http.post<void>(`${this.baseUrl}/auction/${sessionId}/undo`, {});
  }

  /** GET /auction/{sessionId}/projection/{playerId} — live expected price. */
  projection(sessionId: string, playerId: string): Observable<ProjectionResponse> {
    return this.http.get<ProjectionResponse>(
      `${this.baseUrl}/auction/${sessionId}/projection/${playerId}`,
    );
  }

  /**
   * GET /auction/{sessionId}/alternatives/{playerId} — low-cost + closest
   * matches for the given target player, plus WS3 diversified list and bid caps.
   *
   * Optional query params: low_cost_percentile, participant_id, strategy_name.
   */
  alternatives(
    sessionId: string,
    playerId: string,
    req: AlternativesRequest = {},
  ): Observable<AlternativesResponse> {
    let params = new HttpParams();
    if (req.config?.lowCostPercentile != null) {
      params = params.set('low_cost_percentile', String(req.config.lowCostPercentile));
    }
    if (req.participantId) {
      params = params.set('participant_id', req.participantId);
    }
    if (req.strategyName) {
      params = params.set('strategy_name', req.strategyName);
    }
    return this.http.get<AlternativesResponse>(
      `${this.baseUrl}/auction/${sessionId}/alternatives/${playerId}`,
      { params },
    );
  }

  /** GET /auction/{sessionId}/summary — full snapshot of the session. */
  summary(sessionId: string): Observable<AuctionSummary> {
    return this.http.get<AuctionSummary>(
      `${this.baseUrl}/auction/${sessionId}/summary`,
    );
  }

  /**
   * GET /auction/{sessionId}/pool?q=... — lista dei player ancora
   * disponibili per l'asta, opzionalmente filtrata per nome con match
   * substring case-insensitive.
   *
   * Caso d'uso: popolare una dropdown di auto-completamento.  Il client
   * chiama prima questo endpoint (es. con `q='oso'` durante la digitazione),
   * poi usa il `playerId` ricevuto su `projection` / `alternatives`.
   */
  pool(sessionId: string, q?: string): Observable<AuctionPlayerSummary[]> {
    let params = new HttpParams();
    if (q !== undefined && q !== null && q.trim() !== '') {
      params = params.set('q', q.trim());
    }
    return this.http.get<AuctionPlayerSummary[]>(
      `${this.baseUrl}/auction/${sessionId}/pool`,
      { params },
    );
  }

  /** GET /auction/{sessionId}/serialize — opaque payload for save-to-disk. */
  serialize(sessionId: string): Observable<SerializedAuctionStateResponse> {
    return this.http.get<SerializedAuctionStateResponse>(
      `${this.baseUrl}/auction/${sessionId}/serialize`,
    );
  }

  /**
   * POST /auction/deserialize — rebuild a session from a previously
   * serialized payload.  The response carries a fresh `sessionId`.
   */
  deserialize(req: DeserializeAuctionRequest): Observable<InitializeAuctionResponse> {
    return this.http.post<InitializeAuctionResponse>(
      `${this.baseUrl}/auction/deserialize`,
      req,
    );
  }

  /** DELETE /auction/{sessionId} — drop a session (204 on success). */
  discard(sessionId: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/auction/${sessionId}`);
  }

  /** GET /auction/{sessionId}/var-ranking — ESV-ranked available players. */
  varRanking(sessionId: string): Observable<VarRankingResponse> {
    return this.http.get<VarRankingResponse>(
      `${this.baseUrl}/auction/${sessionId}/var-ranking`,
    );
  }
}
