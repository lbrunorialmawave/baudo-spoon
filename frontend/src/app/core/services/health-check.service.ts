import { HttpClient } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { catchError, filter, map, of, switchMap, take, timeout, timer } from 'rxjs';

import { API_BASE_URL } from '../tokens/api-base-url.token';

const POLL_INTERVAL_MS = 3000;
const CHECK_TIMEOUT_MS = 3000;

/** Gates app startup/session on backend availability (Render free tier sleeps when idle). */
@Injectable({ providedIn: 'root' })
export class HealthCheckService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = inject(API_BASE_URL);

  readonly ready = signal(true);

  private polling = false;

  /** Run once at app startup. */
  verify(): void {
    this.recheck();
  }

  /** Run whenever an unrelated API call fails, in case the backend has gone away mid-session. */
  recheck(): void {
    if (this.polling) return;
    this.checkOnce().subscribe(ok => {
      if (!ok) {
        this.ready.set(false);
        this.pollUntilReady();
      }
    });
  }

  private pollUntilReady(): void {
    this.polling = true;
    timer(0, POLL_INTERVAL_MS)
      .pipe(
        switchMap(() => this.checkOnce()),
        filter(ok => ok),
        take(1),
      )
      .subscribe(() => {
        this.polling = false;
        this.ready.set(true);
      });
  }

  /** A slow/hanging request (typical Render cold start) counts as "not ready" too, not just an outright error. */
  private checkOnce() {
    return this.http.get(`${this.apiBaseUrl}/health`).pipe(
      timeout(CHECK_TIMEOUT_MS),
      map(() => true),
      catchError(() => of(false)),
    );
  }
}
