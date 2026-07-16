import { Component, effect, inject, input, output, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { PlayerSeasonStat } from '../../../../core/models/stats.models';
import { PlayerQuotation } from '../../../../core/models/quotations.models';
import { NextSeasonPrediction } from '../../../../core/models/api.models';
import { StatsService } from '../../../../core/services/stats.service';
import { QuotationService } from '../../../../core/services/quotation.service';
import { PredictionService } from '../../../../core/services/prediction.service';
import { SkeletonComponent } from '../../../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../../../shared/components/error-boundary/error-boundary.component';

@Component({
  selector: 'app-player-drawer',
  standalone: true,
  imports: [DecimalPipe, SkeletonComponent, ErrorBoundaryComponent],
  template: `
    <!-- Backdrop -->
    <div class="drawer-backdrop" (click)="closed.emit()"></div>

    <!-- Panel -->
    <aside class="drawer-panel">
      <!-- Header -->
      <div class="drawer-header">
        <div class="min-w-0">
          <h2 class="truncate font-semibold" style="color:var(--color-text-primary)">
            {{ player().player_name }}
          </h2>
          <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">
            {{ player().team_name ?? '—' }} · {{ player().stat_category }}
          </p>
        </div>
        <button class="close-btn" (click)="closed.emit()" aria-label="Close">✕</button>
      </div>

      <div class="drawer-body">
        <!-- Next season prediction -->
        <section class="mb-5">
          <h3 class="section-title">Next Season Forecast</h3>
          @if (nextLoading()) {
            <app-skeleton height="48px" />
          } @else if (nextMlUnavailable()) {
            <app-error-boundary title="Pipeline Not Ready"
              message="ML artifacts are being computed." />
          } @else if (nextPred(); as pred) {
            <div class="rounded-lg p-3" style="background:var(--color-surface)">
              <p class="text-xs" style="color:var(--color-text-secondary)">Predicted fantavoto</p>
              <p class="text-2xl font-bold tabular-nums mt-0.5"
                 style="color:var(--color-accent)">
                {{ pred.predictedNextFantavoto | number:'1.2-2' }}
              </p>
            </div>
          } @else {
            <p class="text-xs" style="color:var(--color-text-secondary)">Not available</p>
          }
        </section>

        <!-- Stats history -->
        <section class="mb-5">
          <h3 class="section-title">Stats History</h3>
          @if (statsLoading()) {
            <div class="space-y-1.5">
              @for (_ of [1,2,3]; track $index) { <app-skeleton height="36px" /> }
            </div>
          } @else if (statsHistory().length) {
            <ul class="space-y-1">
              @for (s of statsHistory(); track s.id) {
                <li class="flex items-center justify-between rounded-lg px-3 py-2"
                    style="background:var(--color-surface)">
                  <span class="text-xs" style="color:var(--color-text-secondary)">
                    {{ s.season.season_label }} · {{ s.stat_category }}
                  </span>
                  <span class="font-mono text-sm font-semibold" style="color:var(--color-accent)">
                    {{ s.value ?? '—' }}
                  </span>
                </li>
              }
            </ul>
          } @else {
            <p class="text-xs" style="color:var(--color-text-secondary)">No data</p>
          }
        </section>

        <!-- Quotation history -->
        <section>
          <h3 class="section-title">Auction Price History</h3>
          @if (quotLoading()) {
            <div class="space-y-1.5">
              @for (_ of [1,2]; track $index) { <app-skeleton height="36px" /> }
            </div>
          } @else if (quotHistory().length) {
            <ul class="space-y-1">
              @for (q of quotHistory(); track q.id) {
                <li class="flex items-center justify-between rounded-lg px-3 py-2"
                    style="background:var(--color-surface)">
                  <span class="text-xs" style="color:var(--color-text-secondary)">
                    {{ q.seasonStart }}/{{ q.seasonStart + 1 }} · {{ q.team }}
                  </span>
                  <div class="flex items-center gap-3 text-xs">
                    <span style="color:var(--color-text-secondary)">
                      qtA <strong style="color:var(--color-text-primary)">{{ q.qtA }}</strong>
                    </span>
                    @if (q.fvm !== null) {
                      <span style="color:var(--color-text-secondary)">
                        fvm <strong style="color:var(--color-text-primary)">{{ q.fvm }}</strong>
                      </span>
                    }
                  </div>
                </li>
              }
            </ul>
          } @else {
            <p class="text-xs" style="color:var(--color-text-secondary)">No quotation data</p>
          }
        </section>
      </div>
    </aside>
  `,
  styles: [`
    :host { display: contents; }
    .drawer-backdrop {
      position: fixed; inset: 0; z-index: 40;
      background: rgba(0,0,0,0.5);
    }
    .drawer-panel {
      position: fixed; right: 0; top: 0; bottom: 0; z-index: 50;
      width: 400px;
      display: flex; flex-direction: column;
      background: var(--color-surface);
      border-left: 1px solid var(--color-border);
      animation: slide-in 180ms ease-out;
    }
    @keyframes slide-in {
      from { transform: translateX(100%); }
      to   { transform: translateX(0); }
    }
    .drawer-header {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
      padding: 16px; border-bottom: 1px solid var(--color-border);
    }
    .close-btn {
      flex-shrink: 0; width: 28px; height: 28px;
      border-radius: 6px; font-size: 12px;
      background: var(--color-surface-raised);
      color: var(--color-text-secondary);
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
    }
    .close-btn:hover { color: var(--color-text-primary); }
    .drawer-body { flex: 1; overflow-y: auto; padding: 16px; }
    .section-title {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.06em; margin-bottom: 8px;
      color: var(--color-text-secondary);
    }
  `],
})
export class PlayerDrawerComponent {
  readonly player = input.required<PlayerSeasonStat>();
  readonly closed = output<void>();

  private readonly statsService = inject(StatsService);
  private readonly quotService = inject(QuotationService);
  private readonly predService = inject(PredictionService);

  readonly statsHistory = signal<PlayerSeasonStat[]>([]);
  readonly quotHistory = signal<PlayerQuotation[]>([]);
  readonly nextPred = signal<NextSeasonPrediction | null>(null);

  readonly statsLoading = signal(false);
  readonly quotLoading = signal(false);
  readonly nextLoading = signal(false);
  readonly nextMlUnavailable = signal(false);

  constructor() {
    effect(() => {
      const p = this.player();

      // Stats history
      this.statsLoading.set(true);
      this.statsService.getPlayerStatsById(p.player_fotmob_id).subscribe({
        next: items => { this.statsHistory.set(items); this.statsLoading.set(false); },
        error: () => this.statsLoading.set(false),
      });

      // Quotation history
      this.quotLoading.set(true);
      this.quotService.getPlayerHistory(p.player_fotmob_id).subscribe({
        next: res => { this.quotHistory.set(res.items); this.quotLoading.set(false); },
        error: () => { this.quotHistory.set([]); this.quotLoading.set(false); },
      });

      // Next season prediction
      this.nextLoading.set(true);
      this.nextMlUnavailable.set(false);
      this.predService.getNextSeason(p.player_name).subscribe({
        next: items => {
          this.nextPred.set(items[0] ?? null);
          this.nextLoading.set(false);
        },
        error: err => {
          if (err.status === 503) this.nextMlUnavailable.set(true);
          this.nextPred.set(null);
          this.nextLoading.set(false);
        },
      });
    });
  }
}
