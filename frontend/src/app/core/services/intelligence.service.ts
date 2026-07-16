import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AlternativesResponse, ClusteringResponse } from '../models/api.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class IntelligenceService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getClusters(): Observable<ClusteringResponse> {
    const params = new HttpParams().set('page', '1').set('size', '400');
    return this.http.get<ClusteringResponse>(`${this.baseUrl}/intelligence/clustering/players`, { params });
  }

  getAlternatives(topPlayerId: number): Observable<AlternativesResponse> {
    const params = new HttpParams().set('top_player_id', topPlayerId);
    return this.http.get<AlternativesResponse>(`${this.baseUrl}/intelligence/clustering/alternatives`, { params });
  }
}
