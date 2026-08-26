import { Component, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { HybridPlayerPrediction } from '../../../../core/models/api.models';
import { FASE7_RENDIMENTO_LABELS, FASE7_PREZZO_LABELS } from '../../../../core/models/mantra.models';

/**
 * Read-only detail panel for a hybrid MANTRA+ML prediction row.
 *
 * Unlike PlayerDrawerComponent, this performs no HTTP calls: every field it
 * shows is already present on the HybridPlayerPrediction row loaded by the
 * Ibrido tab (predictions.component.ts), which has no player-fotmob-id to
 * look up further history with anyway.
 */
@Component({
  selector: 'app-prediction-drawer',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="drawer-backdrop" (click)="closed.emit()"></div>

    <aside class="drawer-panel">
      <div class="drawer-header">
        <div class="min-w-0">
          <h2 class="truncate font-semibold" style="color:var(--color-text-primary)">
            {{ player().playerName ?? '—' }}
          </h2>
          <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">
            {{ player().team ?? '—' }} · {{ player().ruoloPrimario ?? '—' }}
            @if (player().ruoliMantra?.length) {
              ({{ player().ruoliMantra!.join(', ') }})
            }
          </p>
        </div>
        <button class="close-btn" (click)="closed.emit()" aria-label="Close">✕</button>
      </div>

      <div class="drawer-body">
        @if (player().Fase7_Rendimento || player().Fase7_Prezzo || (player().hybridLabels?.length ?? 0) > 0) {
          <section class="mb-5">
            <div class="flex flex-wrap gap-1.5">
              @if (player().Fase7_Rendimento; as f7r) {
                @let f7rMeta = FASE7_RENDIMENTO_LABELS[f7r];
                <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                      [style.background]="f7rMeta?.color ?? '#6B7280'">
                  {{ f7rMeta?.icon ?? '' }} {{ f7rMeta?.label ?? f7r }}
                </span>
              }
              @if (player().Fase7_Prezzo; as f7p) {
                @let f7pMeta = FASE7_PREZZO_LABELS[f7p];
                <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                      [style.background]="f7pMeta?.color ?? '#6B7280'">
                  {{ f7pMeta?.icon ?? '' }} {{ f7pMeta?.label ?? f7p }}
                </span>
              }
              @for (l of player().hybridLabels ?? []; track l) {
                <span class="rounded-full px-2 py-0.5 text-xs font-medium"
                      style="background:var(--color-surface-raised);color:var(--color-text-secondary)">
                  {{ l }}
                </span>
              }
            </div>
          </section>
        }

        <section class="mb-5">
          <h3 class="section-title">Punteggi MANTRA</h3>
          <div class="grid grid-cols-2 gap-2">
            @for (row of mantraRows(); track row.label) {
              <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                  {{ row.value != null ? (row.value | number:'1.1-1') : '—' }}
                </p>
              </div>
            }
          </div>
        </section>

        <section class="mb-5">
          <div class="flex items-center justify-between gap-2 mb-2">
            <h3 class="section-title mb-0">Machine Learning</h3>
            <div class="flex items-center gap-1.5 flex-wrap justify-end">
              @if (player().isForeignFallback) {
                <span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold shrink-0"
                      style="background:#3B82F622;color:#60A5FA;border:1px solid #3B82F644"
                      title="La prediction ML usa statistiche storiche da un campionato estero come fallback.">
                  🌍 Foreign fallback
                </span>
              }
              @if (player().mlValuesNoisy) {
                <span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold shrink-0"
                      style="background:#F59E0B22;color:#FBBF24;border:1px solid #F59E0B44"
                      title="Predizione basata su un campione ridotto di minuti giocati; il valore è stato ammorbidito verso la media di lega.">
                  ⚠️ Valori ML rumorosi
                </span>
              }
            </div>
          </div>
          <div class="grid grid-cols-2 gap-2">
            @for (row of mlRows(); track row.label) {
              <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                  {{ row.value != null ? (row.value | number:'1.1-1') : '—' }}
                </p>
              </div>
            }
          </div>
          @if (!player().hasMlData) {
            <p class="text-xs mt-2" style="color:var(--color-text-secondary)">
              Nessun dato ML disponibile per questo giocatore.
            </p>
          }
        </section>

        <section>
          <h3 class="section-title">Ibrido</h3>
          <div class="grid grid-cols-2 gap-2">
            @for (row of hybridRows(); track row.label) {
              <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                  {{ row.value != null ? (row.value | number:'1.1-1') : '—' }}
                </p>
              </div>
            }
          </div>
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
      width: 100vw;
      display: flex; flex-direction: column;
      background: var(--color-surface);
      border-left: 1px solid var(--color-border);
      animation: slide-in 180ms ease-out;
      padding-bottom: env(safe-area-inset-bottom, 0);
      padding-right: env(safe-area-inset-right, 0);
    }
    @media (min-width: 640px) {
      .drawer-panel { width: 400px; }
    }
    @media (prefers-reduced-motion: reduce) {
      .drawer-panel { animation: none; }
    }
    @keyframes slide-in {
      from { transform: translateX(100%); }
      to   { transform: translateX(0); }
    }
    .drawer-header {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
      padding: 16px; border-bottom: 1px solid var(--color-border);
      padding-top: max(16px, env(safe-area-inset-top, 0));
    }
    .close-btn {
      flex-shrink: 0; width: 44px; height: 44px;
      border-radius: 8px; font-size: 12px;
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
export class PredictionDrawerComponent {
  readonly player = input.required<HybridPlayerPrediction>();
  readonly closed = output<void>();

  readonly FASE7_RENDIMENTO_LABELS = FASE7_RENDIMENTO_LABELS;
  readonly FASE7_PREZZO_LABELS = FASE7_PREZZO_LABELS;

  mantraRows(): { label: string; value: number | null }[] {
    const p = this.player();
    return [
      { label: 'P1', value: p.P1 },
      { label: 'P2', value: p.P2 },
      { label: 'P3', value: p.P3 },
      { label: 'P4', value: p.P4 },
      { label: 'CP', value: p.CP },
      { label: 'FP', value: p.FP },
      { label: 'FP_Mantra', value: p.FP_Mantra },
      { label: 'VR', value: p.VR },
      { label: 'Prezzo Massimo', value: p.Prezzo_Massimo },
    ];
  }

  mlRows(): { label: string; value: number | null }[] {
    const p = this.player();
    return [
      { label: 'Fantavoto previsto', value: p.predictedDisplay ?? p.predictedFantavoto },
      { label: 'Deviazione std.', value: p.predictionStd },
      { label: 'Minuti attesi', value: p.expectedMinutes },
      { label: 'ML score (0-100)', value: p.mlScoreNorm },
      { label: 'ML boost', value: p.mlBoost },
      { label: 'Confidence', value: p.confidenceScore },
    ];
  }

  hybridRows(): { label: string; value: number | null }[] {
    const p = this.player();
    return [
      { label: 'FP Ibrido', value: p.fpIbrido },
      { label: 'Gap MANTRA-ML', value: p.fpGap },
      { label: 'Expected value', value: p.expectedValue },
      { label: 'VAR score', value: p.varScore },
      { label: 'ESV', value: p.esv },
      { label: 'Next season', value: p.nextSeasonPredicted },
    ];
  }
}
