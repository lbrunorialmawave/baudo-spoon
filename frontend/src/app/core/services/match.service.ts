import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { PaginatedResponse } from '../models/api.models';
import { MatchStat } from '../models/stats.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class MatchService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getMatches(opts: {
    league?: string; season?: number; team?: string;
    opponent?: string; search?: string; page?: number; size?: number;
  } = {}): Observable<PaginatedResponse<MatchStat>> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 20);
    if (opts.league)       params = params.set('league', opts.league);
    if (opts.season != null) params = params.set('season', opts.season);
    if (opts.team)         params = params.set('team', opts.team);
    if (opts.opponent)     params = params.set('opponent', opts.opponent);
    if (opts.search)       params = params.set('search', opts.search);
    return this.http.get<PaginatedResponse<MatchStat>>(`${this.baseUrl}/matches/`, { params });
  }

  getMatch(id: number): Observable<MatchStat> {
    return this.http.get<MatchStat>(`${this.baseUrl}/matches/${id}`);
  }
}
