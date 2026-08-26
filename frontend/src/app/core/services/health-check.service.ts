import { HttpClient } from '@angular/common/http';
import { isPlatformBrowser } from '@angular/common';
import { Injectable, PLATFORM_ID, inject, signal } from '@angular/core';
import { Subscription, catchError, filter, map, of, switchMap, take, timeout, timer } from 'rxjs';

import { API_BASE_URL } from '../tokens/api-base-url.token';

const POLL_INTERVAL_MS = 3000;
const CHECK_TIMEOUT_MS = 3000;
/** Well under Render's ~15min free-tier idle timeout, so a ping never arrives too late. */
const KEEP_ALIVE_INTERVAL_MS = 4 * 60 * 1000;
const KEEP_ALIVE_STORAGE_KEY = 'fanta-intelligence.keep-alive';

/** Gates app startup/session on backend availability (Render free tier sleeps when idle). */
@Injectable({ providedIn: 'root' })
export class HealthCheckService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = inject(API_BASE_URL);
  private readonly platformId = inject(PLATFORM_ID);

  readonly ready = signal(true);
  /** User-toggled, persisted in localStorage: keeps pinging the backend while this tab is open,
   *  so it never falls asleep mid-auction. Survives reloads until explicitly turned off. */
  readonly keepAliveEnabled = signal(this.readKeepAliveFlag());

  private polling = false;
  private keepAliveSub?: Subscription;

  constructor() {
    if (this.keepAliveEnabled()) this.startKeepAlive();
  }

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

  toggleKeepAlive(): void {
    this.setKeepAlive(!this.keepAliveEnabled());
  }

  setKeepAlive(enabled: boolean): void {
    this.keepAliveEnabled.set(enabled);
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem(KEEP_ALIVE_STORAGE_KEY, enabled ? '1' : '0');
    }
    if (enabled) this.startKeepAlive();
    else this.stopKeepAlive();
  }

  private startKeepAlive(): void {
    if (this.keepAliveSub || !isPlatformBrowser(this.platformId)) return;
    this.keepAliveSub = timer(KEEP_ALIVE_INTERVAL_MS, KEEP_ALIVE_INTERVAL_MS).subscribe(() => this.recheck());
  }

  private stopKeepAlive(): void {
    this.keepAliveSub?.unsubscribe();
    this.keepAliveSub = undefined;
  }

  private readKeepAliveFlag(): boolean {
    if (!isPlatformBrowser(this.platformId)) return false;
    return localStorage.getItem(KEEP_ALIVE_STORAGE_KEY) === '1';
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
