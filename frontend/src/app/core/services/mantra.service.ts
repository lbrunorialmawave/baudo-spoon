import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';
import {
  MantraPlayer,
  MantraPlayersResponse,
  MantraStatsResponse,
  MantraTopResponse,
  MatchdayPlayerStatus,
  DataHealthResponse,
} from '../models/mantra.models';

@Injectable({ providedIn: 'root' })
export class MantraService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  // ── MANTRA endpoints ───────────────────────────────────────────────────

  /** List all players with MANTRA scores (paginated). */
  listPlayers(opts: {
    ruolo?: string;
    fase7Rendimento?: string;
    fase7Prezzo?: string;
    team?: string;
    search?: string;
    minFp?: number;
    minPrice?: number;
    maxPrice?: number;
    fantacalcioIds?: number[];
    sortBy?: string;
    sortDir?: string;
    page?: number;
    size?: number;
  } = {}): Observable<MantraPlayersResponse> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.ruolo)            params = params.set('ruolo', opts.ruolo);
    if (opts.fase7Rendimento)  params = params.set('fase7_rendimento', opts.fase7Rendimento);
    if (opts.fase7Prezzo)      params = params.set('fase7_prezzo', opts.fase7Prezzo);
    if (opts.team)      params = params.set('team', opts.team);
    if (opts.search)    params = params.set('search', opts.search);
    if (opts.minFp != null)     params = params.set('min_fp', opts.minFp);
    if (opts.minPrice != null)  params = params.set('min_price', opts.minPrice);
    if (opts.maxPrice != null)  params = params.set('max_price', opts.maxPrice);
    if (opts.fantacalcioIds)    params = params.set('fantacalcio_ids', opts.fantacalcioIds.join(','));
    if (opts.sortBy)            params = params.set('sort_by', opts.sortBy);
    if (opts.sortDir)           params = params.set('sort_dir', opts.sortDir);
    return this.http.get<MantraPlayersResponse>(
      `${this.baseUrl}/mantra/players`, { params }
    );
  }

  /** Distinct teams present in the current MANTRA season (matches /mantra/players' data). */
  getTeams(): Observable<{ teams: string[] }> {
    return this.http.get<{ teams: string[] }>(`${this.baseUrl}/mantra/teams`);
  }

  /** Single player detail. */
  getPlayer(fantacalcioId: number): Observable<{ player: MantraPlayer; classifications: any }> {
    return this.http.get<{ player: MantraPlayer; classifications: any }>(
      `${this.baseUrl}/mantra/players/${fantacalcioId}`
    );
  }

  /** Top N per ruolo. */
  getTopPerRuolo(ruolo: string, limit = 15): Observable<MantraTopResponse> {
    const params = new HttpParams().set('limit', limit);
    return this.http.get<MantraTopResponse>(
      `${this.baseUrl}/mantra/top/${ruolo}`, { params }
    );
  }

  /** Classifications overview. */
  getClassifications(): Observable<any> {
    return this.http.get(`${this.baseUrl}/mantra/classifications`);
  }

  /** Run MANTRA computation. */
  runComputation(seasonStart = 2026): Observable<{ status: string; season_start: number; n_players: number }> {
    const params = new HttpParams().set('season_start', seasonStart);
    return this.http.post<{ status: string; season_start: number; n_players: number }>(
      `${this.baseUrl}/mantra/run`, null, { params }
    );
  }

  /** Stats overview. */
  getStats(): Observable<MantraStatsResponse> {
    return this.http.get<MantraStatsResponse>(`${this.baseUrl}/mantra/stats`);
  }

  // ── Matchday endpoints ─────────────────────────────────────────────────

  /** List matchday status for all players. */
  getMatchdayStatus(opts: {
    matchday?: number;
    statusFilter?: string;
    team?: string;
  } = {}): Observable<{ matchday: number; count: number; items: MatchdayPlayerStatus[] }> {
    let params = new HttpParams();
    if (opts.matchday)     params = params.set('matchday', opts.matchday);
    if (opts.statusFilter) params = params.set('status_filter', opts.statusFilter);
    if (opts.team)         params = params.set('team', opts.team);
    return this.http.get<{ matchday: number; count: number; items: MatchdayPlayerStatus[] }>(
      `${this.baseUrl}/matchday/status`, { params }
    );
  }

  /** Consigliati for current matchday. */
  getConsigliati(matchday?: number, minProbability = 70): Observable<{ matchday: number; count: number; items: MatchdayPlayerStatus[] }> {
    let params = new HttpParams().set('min_probability', minProbability);
    if (matchday) params = params.set('matchday', matchday);
    return this.http.get<{ matchday: number; count: number; items: MatchdayPlayerStatus[] }>(
      `${this.baseUrl}/matchday/consigliati`, { params }
    );
  }

  // ── Admin / Data Health ────────────────────────────────────────────────

  /** Data health overview. */
  getDataHealth(): Observable<DataHealthResponse> {
    return this.http.get<DataHealthResponse>(`${this.baseUrl}/admin/data-health`);
  }

  /** Trigger Probabili Formazioni scraper. */
  runProbabiliScraper(matchday?: number): Observable<any> {
    let params = new HttpParams();
    if (matchday) params = params.set('matchday', matchday);
    return this.http.post(`${this.baseUrl}/admin/scrape/probabili`, null, { params });
  }

  /** Trigger Gruppo Esperti ratings scraper. */
  runEspertiScraper(seasonStart?: number): Observable<any> {
    let params = new HttpParams();
    if (seasonStart) params = params.set('season_start', seasonStart);
    return this.http.post(`${this.baseUrl}/admin/scrape/esperti`, null, { params });
  }

  /** Re-import quotazioni XLSX listoni from the mounted ./quotazioni directory. */
  runQuotazioniImport(): Observable<any> {
    return this.http.post(`${this.baseUrl}/admin/scrape/quotazioni`, null);
  }

  /** Fetch career stats for listino players with no Serie A history. */
  runForeignStatsScraper(force = false): Observable<any> {
    const params = new HttpParams().set('force', force);
    return this.http.post(`${this.baseUrl}/admin/scrape/foreign-stats`, null, { params });
  }

  /** Retry FotMob ID resolution for unmatched players, then chain foreign-stats. */
  runResolveUnmatched(): Observable<any> {
    return this.http.post(`${this.baseUrl}/admin/scrape/resolve-unmatched`, null);
  }
}
