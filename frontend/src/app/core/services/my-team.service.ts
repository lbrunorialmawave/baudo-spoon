import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';
import {
  LineupOptimizeRequest,
  LineupOptimizeResponse,
  RosterClaimResponse,
  RosterDetailResponse,
  RosterImportResponse,
  TradesDashboardRequest,
  TradesDashboardResponse,
  TradeEvaluateRequest,
  TradeEvaluateResponse,
  TradeExecuteRequest,
  TradeExecuteResponse,
} from '../models/my-team.models';

@Injectable({ providedIn: 'root' })
export class MyTeamService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  importRoster(file: File, seasonStart?: number): Observable<RosterImportResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    let url = `${this.baseUrl}/roster/import`;
    if (seasonStart != null) {
      url += `?season_start=${seasonStart}`;
    }
    return this.http.post<RosterImportResponse>(url, form);
  }

  claimTeam(
    contextId: string,
    sheetName: string,
    teamName: string,
  ): Observable<RosterClaimResponse> {
    return this.http.post<RosterClaimResponse>(`${this.baseUrl}/roster/claim`, {
      contextId,
      sheetName,
      teamName,
    });
  }

  getTeamRoster(
    contextId: string,
    sheetName: string,
    teamName: string,
  ): Observable<RosterDetailResponse> {
    return this.http.get<RosterDetailResponse>(
      `${this.baseUrl}/roster/context/${encodeURIComponent(contextId)}/team`,
      { params: { sheet_name: sheetName, team_name: teamName } },
    );
  }

  listTeams(
    contextId: string,
    division?: string,
    includeEmpty = false,
  ): Observable<{ contextId: string; teams: import('../models/my-team.models').RosterTeamCard[] }> {
    const params: Record<string, string> = {
      include_empty: String(includeEmpty),
    };
    if (division) {
      params['division'] = division;
    }
    return this.http.get<{ contextId: string; teams: import('../models/my-team.models').RosterTeamCard[] }>(
      `${this.baseUrl}/roster/context/${encodeURIComponent(contextId)}/teams`,
      { params },
    );
  }

  optimizeLineup(req: LineupOptimizeRequest): Observable<LineupOptimizeResponse> {
    return this.http.post<LineupOptimizeResponse>(`${this.baseUrl}/lineup/optimize`, req);
  }

  tradesDashboard(req: TradesDashboardRequest): Observable<TradesDashboardResponse> {
    return this.http.post<TradesDashboardResponse>(`${this.baseUrl}/trades/dashboard`, req);
  }

  executeTrade(req: TradeExecuteRequest): Observable<TradeExecuteResponse> {
    return this.http.post<TradeExecuteResponse>(`${this.baseUrl}/trades/execute`, req);
  }

  evaluateTrade(req: TradeEvaluateRequest): Observable<TradeEvaluateResponse> {
    return this.http.post<TradeEvaluateResponse>(`${this.baseUrl}/trades/evaluate`, req);
  }
}
