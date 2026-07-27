import { Injectable, signal, inject, PLATFORM_ID } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { isPlatformBrowser } from '@angular/common';
import { tap } from 'rxjs/operators';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';

const ACCESS_KEY = 'fanta_access_token';
const REFRESH_KEY = 'fanta_refresh_token';

interface TokenPayload {
  sub: string;
  email: string;
  role: 'admin' | 'member';
  exp: number;
}

interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

interface RefreshResponse {
  access_token: string;
  token_type: string;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly platformId = inject(PLATFORM_ID);
  private readonly baseUrl = inject(API_BASE_URL);
  private readonly endpoint = `${this.baseUrl}/auth/`;
  readonly role = signal<'admin' | 'member' | null>(null);
  readonly isAuthenticated = signal(false);

  constructor() {
    if (isPlatformBrowser(this.platformId)) {
      this._syncFromStorage();
    }
  }

  login(email: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.endpoint}login`, { email, password }).pipe(
      tap(res => {
        localStorage.setItem(ACCESS_KEY, res.access_token);
        localStorage.setItem(REFRESH_KEY, res.refresh_token);
        this._syncFromStorage();
      })
    );
  }

  register(email: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.endpoint}register`, { email, password }).pipe(
      tap(res => {
        localStorage.setItem(ACCESS_KEY, res.access_token);
        localStorage.setItem(REFRESH_KEY, res.refresh_token);
        this._syncFromStorage();
      })
    );
  }

  logout(): Observable<void> {
    const refreshToken = localStorage.getItem(REFRESH_KEY) ?? '';
    return this.http.post<void>(`${this.endpoint}logout`, { refresh_token: refreshToken }).pipe(
      tap(() => this._clear())
    );
  }

  refresh(): Observable<RefreshResponse> {
    const refreshToken = localStorage.getItem(REFRESH_KEY) ?? '';
    return this.http.post<RefreshResponse>(`${this.endpoint}refresh`, { refresh_token: refreshToken }).pipe(
      tap(res => {
        localStorage.setItem(ACCESS_KEY, res.access_token);
        this._syncFromStorage();
      })
    );
  }

  getAccessToken(): string | null {
    return isPlatformBrowser(this.platformId) ? localStorage.getItem(ACCESS_KEY) : null;
  }

  private _syncFromStorage(): void {
    const token = localStorage.getItem(ACCESS_KEY);
    const payload = token ? this._decode(token) : null;
    if (payload && payload.exp * 1000 > Date.now()) {
      this.role.set(payload.role);
      this.isAuthenticated.set(true);
    } else {
      this.role.set(null);
      this.isAuthenticated.set(false);
    }
  }

  private _decode(token: string): TokenPayload | null {
    try {
      return JSON.parse(atob(token.split('.')[1])) as TokenPayload;
    } catch {
      return null;
    }
  }

  private _clear(): void {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    this.role.set(null);
    this.isAuthenticated.set(false);
  }
}
