import { HttpClient } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { catchError, filter, of, switchMap, take, timer } from 'rxjs';

import { API_BASE_URL } from '../tokens/api-base-url.token';

const POLL_INTERVAL_MS = 3000;

/** Gates app startup on backend availability (Render free tier sleeps when idle). */
@Injectable({ providedIn: 'root' })
export class HealthCheckService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = inject(API_BASE_URL);

  readonly ready = signal(true);

  verify(): void {
    this.ping().subscribe({
      error: () => {
        this.ready.set(false);
        this.pollUntilReady();
      },
    });
  }

  private pollUntilReady(): void {
    timer(0, POLL_INTERVAL_MS)
      .pipe(
        switchMap(() => this.ping().pipe(catchError(() => of(null)))),
        filter(result => result !== null),
        take(1),
      )
      .subscribe(() => this.ready.set(true));
  }

  private ping() {
    return this.http.get(`${this.apiBaseUrl}/health`);
  }
}
