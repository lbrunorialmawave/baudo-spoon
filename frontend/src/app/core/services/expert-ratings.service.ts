import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { PlayerExpertRatingsResponse } from '../models/expert-ratings.models';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class ExpertRatingsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  getByFotmobId(playerFotmobId: number): Observable<PlayerExpertRatingsResponse> {
    return this.http.get<PlayerExpertRatingsResponse>(
      `${this.baseUrl}/experts/ratings/by-fotmob/${playerFotmobId}`,
    );
  }
}
