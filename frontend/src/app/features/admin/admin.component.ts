import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { DatePipe, DecimalPipe, PercentPipe } from '@angular/common';
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
  readonly healthError = signal<string | null>(null);
  readonly allOk = signal(false);

  // Latest season derived from /quotations/seasons — avoids hardcoded year.
  readonly currentSeason = signal<number | null>(null);

  constructor() {
    this.quotationService.getSeasons().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (seasons) => {
        const latest = seasons.length ? Math.max(...seasons) : null;
        this.currentSeason.set(latest);
        // Pre-populate the snai scraper param with the current season.
        const snai = this.scrapers.find(s => s.name === 'snai');
        if (snai && latest) snai.params[0].value.set(String(latest));
      },
    });
    this.loadHealth();
    this.loadHybridStatus();
    this.loadHybridConfig();
    this.loadPipelineRuns();
  }

  private loadHealth(): void {
    this.healthError.set(null);
    this.mantraService.getDataHealth().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (res) => {
        this.healthSources.set(res.sources);
        this.allOk.set(res.sources.every(s => s.status === 'ok'));
      },
      error: (e) => this.healthError.set(e?.message ?? 'Failed to load health data'),
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
      description: 'Scrape Serie A winner odds from snai.it',
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
  ];

  readonly runScraper = (scraper: any): void => {
    scraper.running.set(true);
    scraper.result.set(null);
    scraper.error.set(null);

    const obs = scraper.name === 'snai'
      ? this.mantraService.runSnaiScraper(Number(scraper.params[0].value()) || undefined)
      : this.mantraService.runProbabiliScraper(Number(scraper.params[0].value()) || undefined);

    obs.pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (res: any) => {
        scraper.result.set('Scraped ' + (res.records ?? '?') + ' records');
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
    this.predService.runHybrid(this.currentSeason() ?? 2025, overrides as any, true)
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
    this.predService.runHybrid(this.currentSeason() ?? 2025, overrides as any, false)
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
