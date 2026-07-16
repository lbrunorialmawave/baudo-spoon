import { Component, computed, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { PlayerCluster } from '../../../../core/models/api.models';
import { CLUSTER_COLORS } from '../../../../core/constants/cluster-colors';

// Fantacalcio typical range
const SCORE_MIN = 4.5;
const SCORE_MAX = 9.0;

function scorePercent(v: number): number {
  return Math.max(0, Math.min(100, ((v - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100));
}

function scoreTier(v: number): { label: string; color: string } {
  if (v >= 7.5) return { label: 'Top',    color: '#10b981' };
  if (v >= 6.5) return { label: 'Buono',  color: '#22d3ee' };
  if (v >= 5.5) return { label: 'Medio',  color: '#f59e0b' };
  return              { label: 'Basso',  color: '#f43f5e' };
}

@Component({
  selector: 'app-player-card',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="space-y-4" [attr.aria-label]="'Scheda giocatore: ' + player().playerName">

      <!-- ── Identity ─────────────────────────────────────── -->
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0">
          <h3 class="truncate font-semibold" style="color:var(--color-text-primary)">
            {{ player().playerName }}
          </h3>
          <p class="mt-0.5 text-sm" style="color:var(--color-text-secondary)">
            {{ player().teamName ?? 'Squadra sconosciuta' }}
          </p>
        </div>

        <div class="flex shrink-0 flex-col items-end gap-1.5">
          <span
            class="rounded-full px-2.5 py-0.5 text-xs font-semibold text-white"
            [style.background]="clusterColor()"
          >C{{ player().clusterId }}</span>

          @if (player().canonicalRole) {
            <span
              class="rounded-full border px-2 py-0.5 text-xs"
              style="border-color:var(--color-border);color:var(--color-text-secondary)"
            >{{ player().canonicalRole }}</span>
          }
        </div>
      </div>

      <!-- ── Predicted score ──────────────────────────────── -->
      @if (player().predictedFantavoto !== null) {
        <div
          class="rounded-xl p-4"
          style="background:var(--color-surface)"
        >
          <div class="flex items-end justify-between mb-2">
            <div>
              <p class="text-xs font-medium" style="color:var(--color-text-secondary)">
                Fantavoto Previsto
              </p>
              <p class="mt-0.5 text-3xl font-bold tabular-nums"
                 [style.color]="tier().color">
                {{ player().predictedFantavoto! | number:'1.2-2' }}
              </p>
            </div>
            <span
              class="rounded-lg px-2.5 py-1 text-xs font-semibold text-white"
              [style.background]="tier().color"
            >{{ tier().label }}</span>
          </div>

          <!-- Score bar -->
          <div class="relative h-2 overflow-hidden rounded-full"
               style="background:var(--color-surface-raised)"
               role="progressbar"
               [attr.aria-valuenow]="player().predictedFantavoto"
               [attr.aria-valuemin]="4.5"
               [attr.aria-valuemax]="9.0"
               [attr.aria-label]="'Score: ' + player().predictedFantavoto">
            <div
              class="h-full rounded-full transition-all duration-500"
              [style.width.%]="scorePct()"
              [style.background]="tier().color"
            ></div>
          </div>

          <div class="mt-1 flex justify-between text-xs" style="color:var(--color-text-secondary)">
            <span>4.5</span><span>9.0</span>
          </div>
        </div>
      } @else {
        <div class="rounded-xl p-4 text-center text-sm"
             style="background:var(--color-surface);color:var(--color-text-secondary)">
          Previsione non disponibile
        </div>
      }

      <!-- ── PCA Position ─────────────────────────────────── -->
      <div>
        <p class="mb-2 text-xs font-medium" style="color:var(--color-text-secondary)">
          Posizione nello spazio feature
        </p>
        <div class="grid grid-cols-2 gap-3">
          <div class="rounded-lg p-3" style="background:var(--color-surface)">
            <p class="text-xs" style="color:var(--color-text-secondary)">PC 1</p>
            <p class="mt-0.5 font-mono text-sm font-medium"
               style="color:var(--color-text-primary)">
              {{ player().pca0 !== null ? (player().pca0! | number:'1.3-3') : '—' }}
            </p>
          </div>
          <div class="rounded-lg p-3" style="background:var(--color-surface)">
            <p class="text-xs" style="color:var(--color-text-secondary)">PC 2</p>
            <p class="mt-0.5 font-mono text-sm font-medium"
               style="color:var(--color-text-primary)">
              {{ player().pca1 !== null ? (player().pca1! | number:'1.3-3') : '—' }}
            </p>
          </div>
        </div>
      </div>

    </div>
  `,
  styles: [':host { display: block; }'],
})
export class PlayerCardComponent {
  readonly player = input.required<PlayerCluster>();

  readonly clusterColor = computed(() =>
    CLUSTER_COLORS[this.player().clusterId % CLUSTER_COLORS.length]
  );

  readonly tier = computed(() =>
    scoreTier(this.player().predictedFantavoto ?? 0)
  );

  readonly scorePct = computed(() =>
    scorePercent(this.player().predictedFantavoto ?? SCORE_MIN)
  );
}
