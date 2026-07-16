import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { NextSeasonPrediction, PredictionsResponse } from '../models/api.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getPredictions(): Observable<PredictionsResponse> {
    const params = new HttpParams().set('page', '1').set('size', '200');
    return this.http.get<PredictionsResponse>(`${this.baseUrl}/predictions/players`, { params });
  }

  getNextSeason(player?: string): Observable<NextSeasonPrediction[]> {
    let params = new HttpParams();
    if (player) params = params.set('player', player);
    return this.http.get<NextSeasonPrediction[]>(`${this.baseUrl}/predictions/next-season`, { params });
  }
}
