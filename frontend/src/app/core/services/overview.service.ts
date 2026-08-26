import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE_URL } from '../tokens/api-base-url.token';
import { OverviewPlayersResponse } from '../models/overview.models';

@Injectable({ providedIn: 'root' })
export class OverviewService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  /** Paginated, server-side aggregated player list (MANTRA + Hybrid ML +
   *  Gruppo Esperti + titolarità). See GET /overview/players. */
  listPlayers(opts: {
    ruolo?: string;
    team?: string;
    search?: string;
    fase7Rendimento?: string;
    fase7Prezzo?: string;
    labels?: string[];
    confidenceMin?: number;
    minFp?: number;
    maxFp?: number;
    minFpIbrido?: number;
    maxFpIbrido?: number;
    minVr?: number;
    maxVr?: number;
    minPrice?: number;
    maxPrice?: number;
    fantacalcioIds?: number[];
    statusScraped?: string;
    startProbabilityMin?: number;
    probabilityScrapedMin?: number;
    expertTotaleMin?: number;
    expertTotaleMax?: number;
    expertRatingMin?: number;
    expertTitolaritaMin?: number;
    expertMediaVotoMin?: number;
    expertSaluteMin?: number;
    hasMlData?: boolean;
    hasRiskFlag?: boolean;
    /** DRF-style combined sort keys, e.g. 'Pz1,-expert_totale' — already
     *  serialized by the caller (see OverviewComponent.sortByParam). */
    sortBy?: string;
    page?: number;
    size?: number;
  } = {}): Observable<OverviewPlayersResponse> {
    let params = new HttpParams()
      .set('page', opts.page ?? 1)
      .set('size', opts.size ?? 50);
    if (opts.ruolo) params = params.set('ruolo', opts.ruolo);
    if (opts.team) params = params.set('team', opts.team);
    if (opts.search) params = params.set('search', opts.search);
    if (opts.fase7Rendimento) params = params.set('fase7_rendimento', opts.fase7Rendimento);
    if (opts.fase7Prezzo) params = params.set('fase7_prezzo', opts.fase7Prezzo);
    if (opts.labels?.length) params = params.set('labels', opts.labels.join(','));
    if (opts.confidenceMin != null) params = params.set('confidenceMin', opts.confidenceMin);
    if (opts.minFp != null) params = params.set('min_fp', opts.minFp);
    if (opts.maxFp != null) params = params.set('max_fp', opts.maxFp);
    if (opts.minFpIbrido != null) params = params.set('min_fp_ibrido', opts.minFpIbrido);
    if (opts.maxFpIbrido != null) params = params.set('max_fp_ibrido', opts.maxFpIbrido);
    if (opts.minVr != null) params = params.set('min_vr', opts.minVr);
    if (opts.maxVr != null) params = params.set('max_vr', opts.maxVr);
    if (opts.minPrice != null) params = params.set('min_price', opts.minPrice);
    if (opts.maxPrice != null) params = params.set('max_price', opts.maxPrice);
    if (opts.fantacalcioIds) params = params.set('fantacalcio_ids', opts.fantacalcioIds.join(','));
    if (opts.statusScraped) params = params.set('status_scraped', opts.statusScraped);
    if (opts.startProbabilityMin != null) params = params.set('start_probability_min', opts.startProbabilityMin);
    if (opts.probabilityScrapedMin != null) params = params.set('probability_scraped_min', opts.probabilityScrapedMin);
    if (opts.expertTotaleMin != null) params = params.set('expert_totale_min', opts.expertTotaleMin);
    if (opts.expertTotaleMax != null) params = params.set('expert_totale_max', opts.expertTotaleMax);
    if (opts.expertRatingMin != null) params = params.set('expert_rating_min', opts.expertRatingMin);
    if (opts.expertTitolaritaMin != null) params = params.set('expert_titolarita_min', opts.expertTitolaritaMin);
    if (opts.expertMediaVotoMin != null) params = params.set('expert_media_voto_min', opts.expertMediaVotoMin);
    if (opts.expertSaluteMin != null) params = params.set('expert_salute_min', opts.expertSaluteMin);
    if (opts.hasMlData != null) params = params.set('has_ml_data', opts.hasMlData);
    if (opts.hasRiskFlag != null) params = params.set('has_risk_flag', opts.hasRiskFlag);
    if (opts.sortBy) params = params.set('sort_by', opts.sortBy);
    return this.http.get<OverviewPlayersResponse>(`${this.baseUrl}/overview/players`, { params });
  }
}
