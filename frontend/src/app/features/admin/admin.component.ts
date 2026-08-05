import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { DatePipe, DecimalPipe, PercentPipe } from '@angular/common';
import { Subscription, timer } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { MantraService } from '../../core/services/mantra.service';
import { QuotationService } from '../../core/services/quotation.service';
import { PredictionService } from '../../core/services/prediction.service';
import { DataHealthSource } from '../../core/models/mantra.models';
import { HybridConfig, HybridStatus } from '../../core/models/api.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [FormsModule, DatePipe, DecimalPipe, PercentPipe, ErrorBoundaryComponent, SkeletonComponent],
  templateUrl: './admin.component.html',
})
export class AdminComponent {
  private readonly mantraService = inject(MantraService);
  private readonly quotationService = inject(QuotationService);
  private readonly predService = inject(PredictionService);
  private readonly destroyRef = inject(DestroyRef);

  readonly healthSources = signal<DataHealthSource[]>([]);
  readonly healthLoading = signal(true);
  readonly healthError = signal<string | null>(null);
  readonly allOk = signal(false);

  // Latest season derived from /quotations/seasons — avoids hardcoded year.
  readonly currentSeason = signal<number | null>(null);

  constructor() {
    this.quotationService.getSeasons().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (seasons) => {
        const latest = seasons.length ? Math.max(...seasons) : null;
        this.currentSeason.set(latest);
        // Pre-populate the season-scoped scraper params with the current season.
        const snai = this.scrapers.find(s => s.name === 'snai');
        if (snai && latest) snai.params[0].value.set(String(latest));
        const oddsApi = this.scrapers.find(s => s.name === 'odds-api');
        if (oddsApi && latest) oddsApi.params[0].value.set(String(latest));
        const esperti = this.scrapers.find(s => s.name === 'esperti');
        if (esperti && latest) esperti.params[0].value.set(String(latest));
      },
    });
    this.loadHealth();
    this.loadHybridStatus();
    this.loadHybridConfig();
    this.loadPipelineRuns();
    this.loadTrainingStatus();
  }

  private loadHealth(): void {
    this.healthLoading.set(true);
    this.healthError.set(null);
    this.mantraService.getDataHealth().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (res) => {
        this.healthSources.set(res.sources);
        this.allOk.set(res.sources.every(s => s.status === 'ok'));
        this.healthLoading.set(false);
      },
      error: (e) => {
        this.healthError.set(e?.message ?? 'Failed to load health data');
        this.healthLoading.set(false);
      },
    });
  }

  readonly sourceLabel = (name: string): string => {
    const map: Record<string, string> = {
      id_mapping: 'ID Mapping',
      mantra_roles: 'MANTRA Roles',
      matchday_status: 'Probabili Formazioni',
      snai_odds: 'Snai Pre-season Odds',
      expert_ratings: 'Expert Ratings',
      quotations: 'Player Quotations',
    };
    return map[name] ?? name;
  };

  readonly sourceDetail = (s: DataHealthSource): string => {
    const parts: string[] = [];
    if (s.latest_matchday != null) parts.push('Matchday ' + s.latest_matchday);
    if (s.seasons?.length) parts.push('Seasons: ' + s.seasons.join(', '));
    if (!parts.length && s.total_rows) parts.push(s.total_rows + ' rows');
    return parts.join(' · ') || 'No data';
  };

  readonly statusIcon = (status: string): string => {
    if (status === 'ok') return '✅';
    if (status === 'warning') return '⚠️';
    return '❌';
  };

  readonly statusColor = (status: string): string => {
    if (status === 'ok') return '#22C55E';
    if (status === 'warning') return '#F59E0B';
    return '#EF4444';
  };

  readonly scrapers = [
    {
      name: 'snai',
      label: 'Snai Odds',
      description: 'Scrape Serie A winner odds from snai.it (blocked from datacenter IPs)',
      frequency: 'Pre-season + January',
      params: [
        { key: 'season_start', label: 'Season', placeholder: 'current', value: signal('') },
      ],
      running: signal(false),
      result: signal<string | null>(null),
      error: signal<string | null>(null),
    },
    {
      name: 'odds-api',
      label: 'Odds API (Snai replacement)',
      description: 'Scrape Serie A winner odds via the-odds-api.com',
      frequency: 'Pre-season + January',
      params: [
        { key: 'season_start', label: 'Season', placeholder: 'current', value: signal('') },
      ],
      running: signal(false),
      result: signal<string | null>(null),
      error: signal<string | null>(null),
    },
    {
      name: 'probabili',
      label: 'Probabili Formazioni',
      description: 'Scrape probable lineups from fantacalcio.it',
      frequency: 'Every matchday',
      params: [
        { key: 'matchday', label: 'Matchday', placeholder: 'auto', value: signal('') },
      ],
      running: signal(false),
      result: signal<string | null>(null),
      error: signal<string | null>(null),
    },
    {
      name: 'esperti',
      label: 'Gruppo Esperti',
      description: 'Scrape player ratings and comments from forum.gruppoesperti.it',
      frequency: 'Season start + periodic re-scrape',
      params: [
        { key: 'season_start', label: 'Season', placeholder: 'current', value: signal('') },
      ],
      running: signal(false),
      result: signal<string | null>(null),
      error: signal<string | null>(null),
    },
    {
      name: 'quotazioni',
      label: 'Import Quotazioni',
      description: 'Re-import listoni XLSX from the ./quotazioni directory',
      frequency: 'Season start',
      params: [],
      running: signal(false),
      result: signal<string | null>(null),
      error: signal<string | null>(null),
    },
    {
      name: 'foreign-stats',
      label: 'Storico Giocatori Esteri',
      description: 'Recupera lo storico (rating/presenze) per i giocatori del listino senza dati Serie A',
      frequency: 'Dopo ogni import quotazioni',
      params: [
        { key: 'force', label: 'Forza tutti (true/false)', placeholder: 'false', value: signal('') },
      ],
      running: signal(false),
      result: signal<string | null>(null),
      error: signal<string | null>(null),
    },
  ];

  readonly runScraper = (scraper: any): void => {
    scraper.running.set(true);
    scraper.result.set(null);
    scraper.error.set(null);

    const seasonOrMatchday = scraper.params[0] ? Number(scraper.params[0].value()) || undefined : undefined;
    let obs;
    if (scraper.name === 'snai') {
      obs = this.mantraService.runSnaiScraper(seasonOrMatchday);
    } else if (scraper.name === 'odds-api') {
      obs = this.mantraService.runOddsApiScraper(seasonOrMatchday);
    } else if (scraper.name === 'esperti') {
      obs = this.mantraService.runEspertiScraper(seasonOrMatchday);
    } else if (scraper.name === 'quotazioni') {
      obs = this.mantraService.runQuotazioniImport();
    } else if (scraper.name === 'foreign-stats') {
      obs = this.mantraService.runForeignStatsScraper(scraper.params[0].value() === 'true');
    } else {
      obs = this.mantraService.runProbabiliScraper(seasonOrMatchday);
    }

    obs.pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (res: any) => {
        if (scraper.name === 'quotazioni') {
          scraper.result.set(res.status === 'ok' ? 'Import completato' : 'Import fallito: ' + (res.stderr || 'vedi log server'));
        } else if (scraper.name === 'foreign-stats') {
          scraper.result.set(
            `Candidati: ${res.candidates}, recuperati: ${res.fetched}, salvati: ${res.persisted}`
          );
        } else {
          scraper.result.set('Scraped ' + (res.records ?? '?') + ' records');
        }
        scraper.running.set(false);
        this.loadHealth();
      },
      error: (e: any) => {
        scraper.error.set(e?.message ?? 'Scraper failed');
        scraper.running.set(false);
      },
    });
  };

  readonly mantraRunning = signal(false);
  readonly mantraResult = signal<{ season_start: number; n_players: number } | null>(null);

  readonly runMantra = (): void => {
    const season = this.currentSeason() ?? new Date().getFullYear() - 1;
    this.mantraRunning.set(true);
    this.mantraResult.set(null);
    this.mantraService.runComputation(season).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (res) => {
        this.mantraResult.set(res);
        this.mantraRunning.set(false);
      },
      error: () => this.mantraRunning.set(false),
    });
  };

  // ── Hybrid Pipeline ───────────────────────────────────

  readonly hybridStatus = signal<HybridStatus | null>(null);
  readonly hybridConfig = signal<HybridConfig | null>(null);
  readonly hybridLoading = signal(false);
  readonly hybridRunning = signal(false);
  readonly hybridMessage = signal<string | null>(null);
  readonly hybridMessageOk = signal(false);
  readonly lastGenerated = signal<string | null>(null);
  private pendingOverrides: Partial<HybridConfig> = {};

  readonly loadHybridStatus = (): void => {
    this.hybridLoading.set(true);
    this.predService.getHybridStatus().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (s) => { this.hybridStatus.set(s); this.hybridLoading.set(false); },
      error: () => { this.hybridStatus.set(null); this.hybridLoading.set(false); },
    });
  };

  readonly loadHybridConfig = (): void => {
    this.predService.getHybridConfig().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (c) => this.hybridConfig.set(c),
      error: () => {},
    });
  };

  readonly onSliderChange = (event: Event): void => {
    const v = parseInt((event.target as HTMLInputElement).value, 10);
    const current = this.hybridConfig();
    if (current) {
      const updated = { ...current, PESO_MANTRA: v / 100, PESO_ML: (100 - v) / 100 };
      this.hybridConfig.set(updated);
      this.pendingOverrides = { PESO_MANTRA: updated.PESO_MANTRA, PESO_ML: updated.PESO_ML };
    }
  };

  readonly saveAndRegenerate = (): void => {
    const overrides = Object.keys(this.pendingOverrides).length > 0 ? this.pendingOverrides : undefined;
    this.hybridRunning.set(true);
    this.hybridMessage.set(null);
    this.predService.runHybrid(this.currentSeason() ?? new Date().getFullYear() - 1, overrides as any, true)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.lastGenerated.set(res.generatedAt);
          this.hybridMessage.set(`✅ Rigenerato con successo (${res.nPlayers} giocatori).`);
          this.hybridMessageOk.set(true);
          this.hybridRunning.set(false);
          this.pendingOverrides = {};
          this.loadHybridStatus();
          this.loadHybridConfig();
        },
        error: (err) => {
          this.hybridMessage.set('❌ Errore: ' + (err.error?.detail || err.message));
          this.hybridMessageOk.set(false);
          this.hybridRunning.set(false);
        },
      });
  };

  readonly previewConfig = (): void => {
    const overrides = Object.keys(this.pendingOverrides).length > 0 ? this.pendingOverrides : undefined;
    this.hybridRunning.set(true);
    this.hybridMessage.set(null);
    this.predService.runHybrid(this.currentSeason() ?? new Date().getFullYear() - 1, overrides as any, false)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.lastGenerated.set(res.generatedAt);
          this.hybridMessage.set(`✅ Preview generata (${res.nPlayers} giocatori). Non pubblicata.`);
          this.hybridMessageOk.set(true);
          this.hybridRunning.set(false);
        },
        error: (err) => {
          this.hybridMessage.set('❌ Errore: ' + (err.error?.detail || err.message));
          this.hybridMessageOk.set(false);
          this.hybridRunning.set(false);
        },
      });
  };

  // ── ML Pipeline ──────────────────────────────────────

  readonly pipelineRuns = signal<Array<{
    run_id: string;
    model_name: string;
    trained_at: string;
    season_start: number;
    git_commit: string | null;
    status: string;
    metrics: Array<{ metric: string; value: number; split: string }>;
  }> | null>(null);
  readonly pipelineLoading = signal(false);

  readonly loadPipelineRuns = (): void => {
    this.pipelineLoading.set(true);
    this.predService.getPipelineRuns(10, 0)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => { this.pipelineRuns.set(res.items); this.pipelineLoading.set(false); },
        error: () => { this.pipelineRuns.set(null); this.pipelineLoading.set(false); },
      });
  };

  // ── ML Training trigger ───────────────────────────────

  readonly trainingStatus = signal<{
    status: 'idle' | 'running' | 'completed' | 'failed';
    run_number?: number;
    started_at?: string;
    updated_at?: string;
    conclusion?: string | null;
    html_url?: string;
  } | null>(null);
  readonly trainingTriggering = signal(false);
  readonly trainingError = signal<string | null>(null);
  private trainingPollSub: Subscription | null = null;

  readonly loadTrainingStatus = (): void => {
    this.predService.getTrainingStatus().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (s) => {
        this.trainingStatus.set(s);
        if (s.status === 'running') this.startTrainingPoll();
        else this.stopTrainingPoll();
      },
      error: () => {},
    });
  };

  /** @param initialDelayMs first poll delay — a longer grace period is used
   *  right after triggering, since GitHub takes a few seconds to list a
   *  freshly-dispatched run (otherwise the previous run's status would
   *  briefly flash before the new one appears). */
  private startTrainingPoll(initialDelayMs = 5000): void {
    if (this.trainingPollSub) return;
    this.trainingPollSub = timer(initialDelayMs, 5000)
      .pipe(switchMap(() => this.predService.getTrainingStatus()), takeUntilDestroyed(this.destroyRef))
      .subscribe((s) => {
        this.trainingStatus.set(s);
        if (s.status !== 'running') {
          this.stopTrainingPoll();
          this.loadPipelineRuns();
        }
      });
  }

  private stopTrainingPoll(): void {
    this.trainingPollSub?.unsubscribe();
    this.trainingPollSub = null;
  }

  readonly triggerTraining = (): void => {
    this.trainingTriggering.set(true);
    this.trainingError.set(null);
    this.predService.trainModel().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => {
        this.trainingTriggering.set(false);
        // Optimistic: show "running" immediately rather than whatever the
        // previous run's status was, while GitHub registers the new one.
        this.trainingStatus.set({ status: 'running' });
        this.startTrainingPoll(8000);
      },
      error: (err) => {
        this.trainingTriggering.set(false);
        this.trainingError.set(err?.error?.detail || err?.message || 'Avvio training fallito');
      },
    });
  };

  readonly cacheInvalidating = signal(false);
  readonly cacheResult = signal<string | null>(null);

  readonly invalidateCache = (): void => {
    this.cacheInvalidating.set(true);
    this.cacheResult.set(null);
    this.predService.invalidateCache()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.cacheResult.set('✅ ' + (res.detail ?? 'Cache invalidata'));
          this.cacheInvalidating.set(false);
        },
        error: (err) => {
          this.cacheResult.set('❌ Errore: ' + (err.error?.detail || err.message));
          this.cacheInvalidating.set(false);
        },
      });
  };
}
