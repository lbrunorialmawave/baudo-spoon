import { Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { NextSeasonPrediction, PlayerPrediction, ModelComparison } from '../../core/models/api.models';
import { PredictionService } from '../../core/services/prediction.service';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

const SCORE_MIN = 4.5;
const SCORE_MAX = 9.0;

@Component({
  selector: 'app-predictions',
  standalone: true,
  imports: [FormsModule, DecimalPipe, SkeletonComponent, ErrorBoundaryComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Page header -->
      <div class="flex flex-col gap-2 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Predictions</h1>
        <!-- Tab switcher -->
        <div class="flex rounded-lg border p-0.5 self-start sm:self-auto"
             style="border-color:var(--color-border);background:var(--color-surface)">
          @for (tab of tabs; track tab.id) {
            <button
              class="rounded-md px-3 py-1.5 text-xs font-medium transition-colors sm:px-4"
              [style]="selectedTab() === tab.id
                ? 'background:var(--color-accent);color:#fff'
                : 'color:var(--color-text-secondary)'"
              (click)="selectedTab.set(tab.id)">
              {{ tab.label }}
            </button>
          }
        </div>
      </div>

      <!-- ── Current Season tab ──────────────────────────── -->
      @if (selectedTab() === 'current') {
        <div class="p-4 sm:p-6">
          <!-- Model info strip -->
          @if (meta(); as m) {
            <div class="mb-4 rounded-lg border px-4 py-3 text-xs"
                 style="border-color:var(--color-border);background:var(--color-surface)">
              <div class="flex flex-wrap gap-x-6 gap-y-1" style="color:var(--color-text-secondary)">
                <span>Best model: <strong style="color:var(--color-text-primary)">{{ m.bestModel }}</strong></span>
                <span>Run: <code>{{ m.runId.slice(0,12) }}</code></span>
                <span>Role-partitioned: {{ m.rolePartitioned ? 'Yes' : 'No' }}</span>
                @for (mc of m.modelComparison; track mc.model) {
                  <span>{{ mc.model }} R²={{ mc.r2 | number:'1.3-3' }} RMSE={{ mc.rmse | number:'1.2-2' }}</span>
                }
              </div>
            </div>
          }

          <!-- Filters -->
          <div class="mb-4 flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3">
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-55"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);
                          color:var(--color-text-primary)"
                   placeholder="Search player…"
                   [ngModel]="currentSearch()"
                   (ngModelChange)="currentSearch.set($event)" />

            <div class="flex flex-wrap gap-2" role="group">
              <button class="rounded-full border px-3 py-1 text-xs font-medium"
                      [style]="currentRole() === null
                        ? 'background:var(--color-accent);color:#fff;border-color:transparent'
                        : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                      (click)="currentRole.set(null)">All</button>
              @for (role of roles; track role) {
                <button class="rounded-full border px-3 py-1 text-xs font-medium"
                        [style]="currentRole() === role
                          ? 'background:var(--color-accent);color:#fff;border-color:transparent'
                          : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                        (click)="currentRole.set(role)">{{ role }}</button>
              }
            </div>
          </div>

          @if (currentLoading()) {
            <div class="space-y-2">
              @for (_ of skeletonRows; track $index) { <app-skeleton height="52px" /> }
            </div>
          } @else if (currentError()) {
            <app-error-boundary [message]="currentError()!" />
          } @else {
            <div class="card p-0 overflow-hidden">
              <ul class="divide-y" style="--tw-divide-opacity:1">
                @for (p of filteredCurrent(); let i = $index; track p.playerName) {
                  <li class="flex items-center gap-3 px-4 py-3"
                      style="border-color:var(--color-border)">
                    <span class="w-7 text-right font-mono text-xs shrink-0"
                          style="color:var(--color-text-secondary)">{{ i + 1 }}</span>
                    <div class="min-w-0 flex-1">
                      <p class="truncate text-sm font-medium" style="color:var(--color-text-primary)">
                        {{ p.playerName }}
                      </p>
                      <p class="text-xs" style="color:var(--color-text-secondary)">
                        {{ p.teamName ?? '—' }} · {{ p.canonicalRole ?? '—' }}
                      </p>
                    </div>
                    <div class="shrink-0 text-right">
                      @if (p.fantavotoMedio !== null) {
                        <p class="text-xs" style="color:var(--color-text-secondary)">
                          avg {{ p.fantavotoMedio | number:'1.1-1' }}
                        </p>
                      }
                    </div>
                  </li>
                }
              </ul>
            </div>
          }
        </div>
      }

      <!-- ── Next Season tab ─────────────────────────────── -->
      @if (selectedTab() === 'next') {
        <div class="p-4 sm:p-6">
          <div class="mb-4">
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-55"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);
                          color:var(--color-text-primary)"
                   placeholder="Search player…"
                   [ngModel]="nextSearch()"
                   (ngModelChange)="nextSearch.set($event)" />
          </div>

          @if (nextLoading()) {
            <div class="space-y-2">
              @for (_ of skeletonRows; track $index) { <app-skeleton height="52px" /> }
            </div>
          } @else if (nextMlUnavailable()) {
            <app-error-boundary title="Pipeline Not Ready"
              message="Next-season ML artifacts are being computed. Check back in a few minutes." />
          } @else if (nextError()) {
            <app-error-boundary [message]="nextError()!" />
          } @else {
            <div class="card p-0 overflow-hidden">
              <ul class="divide-y">
                @for (p of filteredNext(); let i = $index; track p.playerName) {
                  <li class="flex items-center gap-3 px-4 py-3"
                      style="border-color:var(--color-border)">
                    <span class="w-7 text-right font-mono text-xs shrink-0"
                          style="color:var(--color-text-secondary)">{{ i + 1 }}</span>
                    <p class="min-w-0 flex-1 truncate text-sm font-medium"
                       style="color:var(--color-text-primary)">{{ p.playerName }}</p>
                    <div class="shrink-0 flex items-center gap-3">
                      <!-- Mini score bar -->
                      <div class="w-20 h-1.5 rounded-full overflow-hidden"
                           style="background:var(--color-surface-raised)">
                        <div class="h-full rounded-full"
                             style="background:var(--color-accent)"
                             [style.width.%]="scorePct(p.predictedNextFantavoto)"></div>
                      </div>
                      <span class="font-bold text-sm" style="color:var(--color-accent)">
                        {{ p.predictedNextFantavoto | number:'1.2-2' }}
                      </span>
                    </div>
                  </li>
                }
              </ul>
            </div>
            <!-- ponytail: no role on NextSeasonPrediction — add filter when API enriches the field -->
          }
        </div>
      }
    </div>
  `,
})
export class PredictionsComponent {
  private readonly predService = inject(PredictionService);

  readonly tabs = [
    { id: 'current' as const, label: 'Current Season' },
    { id: 'next' as const, label: 'Next Season' },
  ];
  readonly roles = ['GK', 'DEF', 'MID', 'FWD'];
  readonly skeletonRows = Array.from({ length: 8 });
  readonly selectedTab = signal<'current' | 'next'>('current');

  // Current season
  readonly currentItems = signal<PlayerPrediction[]>([]);
  readonly meta = signal<{ runId: string; bestModel: string; rolePartitioned: boolean; modelComparison: ModelComparison[] } | null>(null);
  readonly currentLoading = signal(true);
  readonly currentError = signal<string | null>(null);
  readonly currentSearch = signal('');
  readonly currentRole = signal<string | null>(null);

  readonly filteredCurrent = computed(() => {
    const q = this.currentSearch().toLowerCase();
    const role = this.currentRole();
    return this.currentItems()
      .filter(p => (!q || p.playerName.toLowerCase().includes(q)) &&
                   (!role || p.canonicalRole === role))
      .sort((a, b) => b.predicted - a.predicted);
  });

  // Next season
  readonly nextItems = signal<NextSeasonPrediction[]>([]);
  readonly nextLoaded = signal(false);
  readonly nextLoading = signal(false);
  readonly nextError = signal<string | null>(null);
  readonly nextMlUnavailable = signal(false);
  readonly nextSearch = signal('');

  readonly filteredNext = computed(() => {
    const q = this.nextSearch().toLowerCase();
    return this.nextItems()
      .filter(p => !q || p.playerName.toLowerCase().includes(q))
      .sort((a, b) => b.predictedNextFantavoto - a.predictedNextFantavoto);
  });

  scorePct(v: number): number {
    return Math.max(0, Math.min(100, ((v - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100));
  }

  constructor() {
    // Load current season on init
    this.predService.getPredictions().subscribe({
      next: res => {
        this.currentItems.set(res.items);
        this.meta.set({ runId: res.runId, bestModel: res.bestModel, rolePartitioned: res.rolePartitioned, modelComparison: res.modelComparison });
        this.currentLoading.set(false);
      },
      error: () => {
        this.currentError.set('Could not load predictions.');
        this.currentLoading.set(false);
      },
    });

    // Load next season lazily when tab selected
    effect(() => {
      if (this.selectedTab() !== 'next' || this.nextLoaded()) return;
      this.nextLoading.set(true);
      this.predService.getNextSeason().subscribe({
        next: items => {
          this.nextItems.set(items);
          this.nextLoaded.set(true);
          this.nextLoading.set(false);
        },
        error: err => {
          if (err.status === 503) this.nextMlUnavailable.set(true);
          else this.nextError.set(err.status === 404 ? 'No next-season predictions available yet.' : 'Could not load predictions.');
          this.nextLoading.set(false);
        },
      });
    });
  }
}
