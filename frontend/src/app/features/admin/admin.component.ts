import { Component, inject, signal } from '@angular/core';
import { DecimalPipe, DatePipe } from '@angular/common';
import { MantraService } from '../../core/services/mantra.service';
import { DataHealthSource } from '../../core/models/mantra.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [DecimalPipe, DatePipe, ErrorBoundaryComponent],
  templateUrl: './admin.component.html',
})
export class AdminComponent {
  private readonly mantraService = inject(MantraService);

  readonly healthSources = signal<DataHealthSource[]>([]);
  readonly healthError = signal<string | null>(null);
  readonly allOk = signal(false);

  constructor() {
    this.loadHealth();
  }

  private loadHealth(): void {
    this.healthError.set(null);
    this.mantraService.getDataHealth().subscribe({
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
    return parts.join(' \u00B7 ') || 'No data';
  };

  readonly statusIcon = (status: string): string => {
    if (status === 'ok') return '\u2705';
    if (status === 'warning') return '\u26A0\uFE0F';
    return '\u274C';
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
        { key: 'season_start', label: 'Season', placeholder: '2025', value: signal('2025') },
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

    obs.subscribe({
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
    this.mantraRunning.set(true);
    this.mantraResult.set(null);
    this.mantraService.runComputation(2025).subscribe({
      next: (res) => {
        this.mantraResult.set(res);
        this.mantraRunning.set(false);
      },
      error: () => this.mantraRunning.set(false),
    });
  };
}
