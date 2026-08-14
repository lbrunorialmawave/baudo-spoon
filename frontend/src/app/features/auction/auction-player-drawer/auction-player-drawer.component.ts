import { Component, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';

/**
 * Minimal view-model for the auction player drawer.
 * Built from VarRankingItem, AuctionPlayerSummary, or assignment rows —
 * only fields already present on the clicked record (no extra HTTP).
 */
export interface AuctionDrawerPlayer {
  playerId: string;
  name: string;
  role: string;
  realTeam?: string | null;
  cost?: number | null;
  projectedScore?: number | null;
  varScore?: number | null;
  expectedPrice?: number | null;
  esv?: number | null;
  calibrated?: boolean | null;
  buySignal?: boolean | null;
  seasonValue?: number | null;
  startProbability?: number | null;
  /** Present when opened from assignment history. */
  finalPrice?: number | null;
  tier?: string | null;
  /** PR9: INSUFFICIENT | LIMITED | STANDARD */
  sampleCohort?: string | null;
  reliabilityWeight?: number | null;
}

/**
 * Read-only detail panel for an auction player record.
 *
 * Purely presentational: shows only data already on the row that was
 * clicked (VAR ranking, pool suggestion, or assignment history).
 */
@Component({
  selector: 'app-auction-player-drawer',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="drawer-backdrop" (click)="closed.emit()"></div>

    <aside class="drawer-panel" role="dialog" aria-modal="true" [attr.aria-label]="'Dettaglio ' + (player().name || 'giocatore')">
      <div class="drawer-header">
        <div class="min-w-0">
          <h2 class="truncate font-semibold" style="color:var(--color-text-primary)">
            {{ player().name || '—' }}
          </h2>
          <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">
            {{ player().realTeam || '—' }} · {{ player().role || '—' }}
            @if (player().buySignal === true) {
              <span class="signal-buy"> · COMPRA</span>
            } @else if (player().buySignal === false) {
              <span class="signal-hold"> · —</span>
            }
          </p>
          @if (isNoisyCohort()) {
            <span class="ml-noisy-badge"
                  [attr.title]="cohortTooltip()">
              ⚠️ Valori ML rumorosi · {{ player().sampleCohort }}
            </span>
          }
        </div>
        <button type="button" class="close-btn" (click)="closed.emit()" aria-label="Chiudi">✕</button>
      </div>

      <div class="drawer-body">
        <section class="mb-5">
          <h3 class="section-title">Identità</h3>
          <div class="grid grid-cols-2 gap-2">
            @for (row of identityRows(); track row.label) {
              <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                <p class="font-mono text-sm font-semibold truncate" style="color:var(--color-text-primary)">
                  {{ row.value }}
                </p>
              </div>
            }
          </div>
        </section>

        @if (hasRanking()) {
          <section class="mb-5">
            <h3 class="section-title">Ranking VAR / ESV</h3>
            <div class="grid grid-cols-2 gap-2">
              @for (row of rankingRows(); track row.label) {
                <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                  <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                  <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                    {{ row.display }}
                  </p>
                </div>
              }
            </div>
          </section>
        }

        @if (hasAssignment()) {
          <section class="mb-5">
            <h3 class="section-title">Assegnazione</h3>
            <div class="grid grid-cols-2 gap-2">
              @for (row of assignmentRows(); track row.label) {
                <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                  <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                  <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                    {{ row.display }}
                  </p>
                </div>
              }
            </div>
          </section>
        }

        <section>
          <h3 class="section-title">Proiezione base</h3>
          <div class="grid grid-cols-2 gap-2">
            @for (row of baseRows(); track row.label) {
              <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
                <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                  {{ row.display }}
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
      cursor: pointer; border: none;
    }
    .close-btn:hover { color: var(--color-text-primary); }
    .drawer-body { flex: 1; overflow-y: auto; padding: 16px; }
    .section-title {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.06em; margin-bottom: 8px;
      color: var(--color-text-secondary);
    }
    .signal-buy { color: var(--color-success, #22C55E); font-weight: 600; }
    .signal-hold { color: var(--color-text-secondary); }
    .ml-noisy-badge {
      display: inline-flex; align-items: center; gap: 4px;
      margin-top: 6px; padding: 2px 8px; border-radius: 999px;
      font-size: 10px; font-weight: 700;
      background: #F59E0B22; color: #FBBF24; border: 1px solid #F59E0B44;
    }
  `],
})
export class AuctionPlayerDrawerComponent {
  readonly player = input.required<AuctionDrawerPlayer>();
  readonly closed = output<void>();

  isNoisyCohort(): boolean {
    const c = this.player().sampleCohort;
    return c === 'LIMITED' || c === 'INSUFFICIENT';
  }

  cohortTooltip(): string {
    const c = this.player().sampleCohort;
    if (c === 'INSUFFICIENT') {
      return 'Campione insufficiente (<100 min): predizione fortemente ammorbidita verso la media di ruolo.';
    }
    return 'Campione limitato (100–799 min): predizione ammorbidita verso la media di ruolo.';
  }

  identityRows(): { label: string; value: string }[] {
    const p = this.player();
    const rows = [
      { label: 'ID', value: p.playerId || '—' },
      { label: 'Ruolo', value: p.role || '—' },
      { label: 'Squadra', value: p.realTeam || '—' },
      { label: 'Nome', value: p.name || '—' },
    ];
    if (p.sampleCohort) {
      rows.push({ label: 'Sample cohort', value: p.sampleCohort });
    }
    if (p.reliabilityWeight != null) {
      rows.push({ label: 'Reliability weight', value: String(p.reliabilityWeight) });
    }
    return rows;
  }

  hasRanking(): boolean {
    const p = this.player();
    return p.varScore != null || p.esv != null || p.expectedPrice != null;
  }

  rankingRows(): { label: string; display: string }[] {
    const p = this.player();
    const fmt = (v: number | null | undefined, digits = '1.1-1') =>
      v != null && Number.isFinite(v) ? v.toLocaleString('it-IT', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) : '—';
    return [
      { label: 'ESV', display: fmt(p.esv) },
      { label: 'Prezzo atteso', display: p.expectedPrice != null ? Math.round(p.expectedPrice).toString() : '—' },
      { label: 'VAR score', display: fmt(p.varScore) },
      {
        label: 'Season value',
        display: p.seasonValue != null ? fmt(p.seasonValue) : '—',
      },
      {
        label: 'Start %',
        display:
          p.startProbability != null
            ? `${Math.round(p.startProbability * 100)}%`
            : '—',
      },
      {
        label: 'Segnale',
        display: p.buySignal === true ? 'COMPRA' : p.buySignal === false ? '—' : '—',
      },
      {
        label: 'Calibrato',
        display: p.calibrated === true ? 'Sì' : p.calibrated === false ? 'No' : '—',
      },
    ];
  }

  hasAssignment(): boolean {
    const p = this.player();
    return p.finalPrice != null || p.tier != null;
  }

  assignmentRows(): { label: string; display: string }[] {
    const p = this.player();
    return [
      { label: 'Prezzo finale', display: p.finalPrice != null ? String(p.finalPrice) : '—' },
      { label: 'Tier', display: p.tier || '—' },
    ];
  }

  baseRows(): { label: string; display: string }[] {
    const p = this.player();
    const fmt2 = (v: number | null | undefined) =>
      v != null && Number.isFinite(v)
        ? v.toLocaleString('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
        : '—';
    return [
      { label: 'Costo listino', display: p.cost != null ? String(p.cost) : '—' },
      { label: 'Score proiettato', display: fmt2(p.projectedScore) },
    ];
  }
}
