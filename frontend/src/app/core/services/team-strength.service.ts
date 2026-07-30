import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, shareReplay } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';

@Injectable({ providedIn: 'root' })
export class TeamStrengthService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);
  private cache$: Observable<Record<string, number>> | null = null;

  getScores(): Observable<Record<string, number>> {
    if (!this.cache$) {
      this.cache$ = this.http.get<Record<string, number>>(
        `${this.baseUrl}/optimize/team-strength`
      ).pipe(shareReplay(1));
    }
    return this.cache$;
  }
}
