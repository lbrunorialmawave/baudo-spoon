import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { PaginatedResponse } from '../models/api.models';
import { PlayerSeasonStat, TeamSeasonStat } from '../models/stats.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class StatsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getPlayerCategories(league?: string, season?: number): Observable<string[]> {
    let params = new HttpParams();
    if (league) params = params.set('league', league);
    if (season != null) params = params.set('season', season);
    return this.http.get<string[]>(`${this.baseUrl}/stats/players/categories`, { params });
  }

  getPlayerStats(opts: {
    league?: string; season?: number; statCategory?: string;
    player?: string; team?: string; page?: number; size?: number;
  } = {}): Observable<PaginatedResponse<PlayerSeasonStat>> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.league)       params = params.set('league', opts.league);
    if (opts.season != null) params = params.set('season', opts.season);
    if (opts.statCategory) params = params.set('stat_category', opts.statCategory);
    if (opts.player)       params = params.set('player', opts.player);
    if (opts.team)         params = params.set('team', opts.team);
    return this.http.get<PaginatedResponse<PlayerSeasonStat>>(`${this.baseUrl}/stats/players`, { params });
  }

  getPlayerStatsById(playerId: number, league?: string): Observable<PlayerSeasonStat[]> {
    let params = new HttpParams();
    if (league) params = params.set('league', league);
    return this.http.get<PlayerSeasonStat[]>(`${this.baseUrl}/stats/players/${playerId}`, { params });
  }

  getTeamCategories(league?: string, season?: number): Observable<string[]> {
    let params = new HttpParams();
    if (league) params = params.set('league', league);
    if (season != null) params = params.set('season', season);
    return this.http.get<string[]>(`${this.baseUrl}/stats/teams/categories`, { params });
  }

  getTeamStats(opts: {
    league?: string; season?: number; statCategory?: string;
    team?: string; page?: number; size?: number;
  } = {}): Observable<PaginatedResponse<TeamSeasonStat>> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.league)       params = params.set('league', opts.league);
    if (opts.season != null) params = params.set('season', opts.season);
    if (opts.statCategory) params = params.set('stat_category', opts.statCategory);
    if (opts.team)         params = params.set('team', opts.team);
    return this.http.get<PaginatedResponse<TeamSeasonStat>>(`${this.baseUrl}/stats/teams`, { params });
  }

  getTeamStatsById(teamId: number): Observable<TeamSeasonStat[]> {
    return this.http.get<TeamSeasonStat[]>(`${this.baseUrl}/stats/teams/${teamId}`);
  }
}
