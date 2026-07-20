import { Component, computed, effect, inject, signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { DecimalPipe } from '@angular/common';
import {
  ClusteringResponse,
  LowCostAlternative,
  ModelComparison,
  PlayerCluster,
  PlayerPrediction,
} from '../../core/models/api.models';
import { IntelligenceService } from '../../core/services/intelligence.service';
import { PredictionService } from '../../core/services/prediction.service';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { PcaScatterComponent } from './components/pca-scatter/pca-scatter.component';
import { PlayerCardComponent } from './components/player-card/player-card.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    SkeletonComponent,
    ErrorBoundaryComponent,
    DecimalPipe,
    PcaScatterComponent,
    PlayerCardComponent,
  ],
  template: `
    <div style="background:var(--color-bg)">
      <!-- ── Page header ──────────────────────────────────── -->
      <div class="flex flex-col gap-2 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Dashboard</h1>
        <!-- Cluster meta chips -->
        @if (clusterData(); as cd) {
          <div class="flex flex-wrap items-center gap-1.5 text-xs font-mono sm:gap-2"
               style="color:var(--color-text-secondary)">
            <span class="rounded px-2 py-1" style="background:var(--color-surface-raised)">
              {{ cd.items.length }} giocatori
            </span>
            <span class="rounded px-2 py-1" style="background:var(--color-surface-raised)">
              {{ cd.clusteringStats.nClusters }} cluster
            </span>
            @if (cd.clusteringStats.silhouette !== null) {
              <span class="rounded px-2 py-1" style="background:var(--color-surface-raised)">
                sil {{ cd.clusteringStats.silhouette | number:'1.3-3' }}
              </span>
            }
            @if (varianceExplained(); as v) {
              <span class="rounded px-2 py-1" style="background:var(--color-surface-raised)">
                {{ v }}% var
              </span>
            }
          </div>
        } @else {
          <app-skeleton height="28px" />
        }
      </div>

      <!-- ── Main grid ─────────────────────────────────────── -->
      <main class="grid grid-cols-1 gap-4 p-4 sm:gap-6 sm:p-6 xl:grid-cols-3">

        <!-- ── PCA Scatter (2/3) ──────────────────────────── -->
        <section class="card p-3 sm:p-4 xl:col-span-2" aria-label="Cluster Map">
          <div class="mb-1 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <h2 class="font-semibold" style="color:var(--color-text-primary)">Cluster Map</h2>
            <p class="text-xs" style="color:var(--color-text-secondary)">
              <span class="hidden sm:inline">Scroll to zoom · Drag to pan · Click to inspect</span>
              <span class="sm:hidden">Tap to inspect</span>
            </p>
          </div>

          <!-- Model stats strip -->
          @if (predictionsMeta(); as meta) {
            <div class="mb-3 rounded-lg border px-3 py-2"
                 style="border-color:var(--color-border);background:var(--color-surface)">
              <button class="flex w-full items-center justify-between text-xs"
                      style="color:var(--color-text-secondary)"
                      (click)="statsStripOpen.update(v => !v)">
                <span>
                  Model: <strong style="color:var(--color-text-primary)">{{ meta.bestModel }}</strong>
                  · Run <code>{{ meta.runId.slice(0, 8) }}</code>
                </span>
                <span>{{ statsStripOpen() ? '▲' : '▼' }}</span>
              </button>
              @if (statsStripOpen()) {
                <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs"
                     style="color:var(--color-text-secondary)">
                  @for (m of meta.modelComparison; track m.model) {
                    <span>{{ m.model }} R²={{ m.r2 | number:'1.3-3' }}</span>
                  }
                </div>
              }
            </div>
          }

          <!-- Role filter chips -->
          @if (availableRoles().length) {
            <div class="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filtra per ruolo">
              <button
                class="rounded-full border px-3 py-1 text-xs font-medium transition-colors"
                [style]="selectedRoles().size === 0
                  ? 'background:var(--color-accent);color:#fff;border-color:transparent'
                  : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                (click)="clearRoles()"
              >Tutti</button>

              @for (role of availableRoles(); track role) {
                <button
                  class="rounded-full border px-3 py-1 text-xs font-medium transition-colors"
                  [style]="selectedRoles().has(role)
                    ? 'background:var(--color-accent);color:#fff;border-color:transparent'
                    : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                  (click)="toggleRole(role)"
                  [attr.aria-pressed]="selectedRoles().has(role)"
                >{{ role }}</button>
              }
            </div>
          }

          @if (!clusterData()) {
            <app-skeleton height="400px" />
          } @else {
            <app-pca-scatter
              [players]="filteredPlayers()"
              (playerSelected)="onPlayerSelect($event)"
            />
          }
        </section>

        <!-- ── Right sidebar (1/3) ───────────────────────── -->
        <section class="card" aria-live="polite" aria-label="Detail panel">
          @if (selectedPlayer(); as player) {
            <!-- Player detail view -->
            <div class="mb-4 flex items-center justify-between">
              <h2 class="font-semibold" style="color:var(--color-text-primary)">Dettaglio</h2>
              <button
                class="flex h-7 w-7 items-center justify-center rounded-lg text-xs transition hover:opacity-80"
                style="background:var(--color-surface);color:var(--color-text-secondary)"
                (click)="selectedPlayer.set(null)"
                aria-label="Chiudi dettaglio"
              >✕</button>
            </div>
            <app-player-card [player]="player" />

            <!-- Budget alternatives -->
            @if (alternativesLoading()) {
              <div class="mt-4 space-y-2">
                @for (_ of [1,2,3]; track $index) {
                  <app-skeleton height="48px" />
                }
              </div>
            } @else if (alternativesMlUnavailable()) {
              <app-error-boundary class="mt-4"
                title="Pipeline Not Ready"
                message="ML artifacts are being computed. Check back in a few minutes." />
            } @else if (alternativesError()) {
              <app-error-boundary class="mt-4" [message]="alternativesError()!" />
            } @else if (alternatives().length) {
              <div class="mt-5">
                <h3 class="mb-2 text-xs font-semibold uppercase tracking-wide"
                    style="color:var(--color-text-secondary)">Budget Alternatives</h3>
                <ul class="space-y-1.5">
                  @for (alt of alternatives(); track alt.altPlayerName) {
                    <li class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                      <p class="text-sm font-medium" style="color:var(--color-text-primary)">
                        {{ alt.altPlayerName }}
                      </p>
                      <p class="text-xs" style="color:var(--color-text-secondary)">
                        {{ alt.altPlayerTeam ?? '—' }}
                        @if (alt.altPlayerFantavoto !== null) {
                          · ★ {{ alt.altPlayerFantavoto | number:'1.1-1' }}
                        }
                      </p>
                    </li>
                  }
                </ul>
              </div>
            }

          } @else {
            <!-- Predictions list -->
            <h2 class="mb-4 font-semibold" style="color:var(--color-text-primary)">
              Top Fantavoto Medio
            </h2>

            @if (predictionsLoading()) {
              <div class="space-y-3">
                @for (_ of placeholders; track $index) {
                  <app-skeleton height="56px" />
                }
              </div>
            } @else if (predictionsError()) {
              <app-error-boundary [message]="predictionsError()!" />
            } @else {
              <ul class="space-y-2" role="list">
                @for (p of topPredictions(); track p.playerName) {
                  <li class="flex cursor-default items-center justify-between rounded-lg px-3 py-2.5 transition hover:opacity-90"
                      style="background:var(--color-surface)">
                    <div class="min-w-0">
                      <p class="truncate text-sm font-medium" style="color:var(--color-text-primary)">
                        {{ p.playerName }}
                      </p>
                      <p class="text-xs" style="color:var(--color-text-secondary)">
                        {{ p.teamName ?? '—' }} · {{ p.canonicalRole ?? '—' }}
                      </p>
                    </div>
                    <div class="shrink-0 text-right">
                      <p class="text-sm font-bold" style="color:var(--color-accent)">
                        {{ p.fantavotoMedio ?? 0 | number: '1.1-1' }}
                      </p>
                    </div>
                  </li>
                }
              </ul>
            }
          }
        </section>
      </main>
    </div>
  `,
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly predictionService = inject(PredictionService);
  private readonly intelligenceService = inject(IntelligenceService);

  // ── Cluster data (from resolver) ────────────────────────
  readonly clusterData = signal<ClusteringResponse | null>(this.route.snapshot.data['clusterData']);

  readonly varianceExplained = computed(() => {
    const cd = this.clusterData();
    if (!cd?.clusteringStats.pcaExplainedVariance?.length) return null;
    const total = cd.clusteringStats.pcaExplainedVariance.reduce((a, b) => a + b, 0);
    return (total * 100).toFixed(1);
  });

  // ── Role filters ─────────────────────────────────────────
  readonly selectedRoles = signal<Set<string>>(new Set());

  readonly availableRoles = computed(() => {
    const items = this.clusterData()?.items ?? [];
    const roles = [...new Set(items.map(p => p.canonicalRole).filter(Boolean))] as string[];
    const order = ['P', 'D', 'C', 'A', 'GK', 'DEF', 'MID', 'FWD'];
    return roles.sort((a, b) => {
      const ai = order.indexOf(a), bi = order.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      return a.localeCompare(b);
    });
  });

  readonly filteredPlayers = computed(() => {
    const items = this.clusterData()?.items ?? [];
    const roles = this.selectedRoles();
    return roles.size === 0 ? items : items.filter(p => p.canonicalRole && roles.has(p.canonicalRole));
  });

  toggleRole(role: string): void {
    this.selectedRoles.update(prev => {
      const next = new Set(prev);
      next.has(role) ? next.delete(role) : next.add(role);
      return next;
    });
  }

  clearRoles(): void { this.selectedRoles.set(new Set()); }

  // ── Player selection ────────────────────────────────────
  readonly selectedPlayer = signal<PlayerCluster | null>(null);

  onPlayerSelect(player: PlayerCluster): void {
    this.selectedPlayer.update(prev => prev?.playerName === player.playerName ? null : player);
  }

  // ── Predictions ─────────────────────────────────────────
  readonly predictionsLoading = signal(true);
  readonly predictionsError = signal<string | null>(null);
  readonly predictions = signal<PlayerPrediction[]>([]);
  readonly predictionsMeta = signal<{ runId: string; bestModel: string; modelComparison: ModelComparison[] } | null>(null);
  readonly statsStripOpen = signal(false);
  readonly placeholders = Array.from({ length: 5 });

  readonly topPredictions = computed(() =>
    this.predictions()
      .slice()
      .sort((a, b) => (b.fantavotoMedio ?? 0) - (a.fantavotoMedio ?? 0))
      .slice(0, 10)
  );

  // ── Budget alternatives ─────────────────────────────────
  readonly alternatives = signal<LowCostAlternative[]>([]);
  readonly alternativesLoading = signal(false);
  readonly alternativesError = signal<string | null>(null);
  readonly alternativesMlUnavailable = signal(false);

  constructor() {
    effect(() => {
      const player = this.selectedPlayer();
      if (!player?.playerFotmobId) {
        this.alternatives.set([]);
        return;
      }
      this.alternativesLoading.set(true);
      this.alternativesError.set(null);
      this.alternativesMlUnavailable.set(false);
      this.intelligenceService.getAlternatives(player.playerFotmobId).subscribe({
        next: res => {
          this.alternatives.set(res.lowCostRecommendations.slice(0, 5));
          this.alternativesLoading.set(false);
        },
        error: err => {
          if (err.status === 503) this.alternativesMlUnavailable.set(true);
          else this.alternativesError.set('Could not load alternatives.');
          this.alternativesLoading.set(false);
        },
      });
    });
  }

  ngOnInit(): void {
    this.predictionService.getPredictions().subscribe({
      next: res => {
        this.predictions.set(res.items ?? []);
        this.predictionsMeta.set({ runId: res.runId, bestModel: res.bestModel, modelComparison: res.modelComparison });
        this.predictionsLoading.set(false);
      },
      error: () => {
        this.predictionsError.set("Impossibile caricare le previsioni. Controlla la connessione all'API.");
        this.predictionsLoading.set(false);
      },
    });
  }
}
