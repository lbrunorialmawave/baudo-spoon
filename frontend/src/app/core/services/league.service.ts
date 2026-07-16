import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { League, Season } from '../models/stats.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class LeagueService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getLeagues(): Observable<League[]> {
    return this.http.get<League[]>(`${this.baseUrl}/leagues/`);
  }

  getSeasons(league?: string): Observable<Season[]> {
    let params = new HttpParams();
    if (league) params = params.set('league', league);
    return this.http.get<Season[]>(`${this.baseUrl}/seasons/`, { params });
  }
}
