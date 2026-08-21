import { DecimalPipe, NgClass } from '@angular/common';
import {
  Component,
  DestroyRef,
  Input,
  OnChanges,
  SimpleChanges,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { finalize } from 'rxjs/operators';
import { MyTeamService } from '../../../core/services/my-team.service';
import {
  TradeConfidence,
  TradeEvaluateResponse,
  TradeMode,
  TradePlayerPtvView,
  TradeVerdict,
  TradesDashboardResponse,
} from '../../../core/models/my-team.models';

/** Selectable player row for the Cedo / Ricevo columns. */
export interface EvaluatorPlayerOption {
  playerId: string;
  name: string;
  roles: string[];
  fpCorr?: number;
  teamSerieA?: string;
  side: 'own' | 'market';
}

@Component({
  selector: 'app-trade-evaluator',
  standalone: true,
  imports: [FormsModule, DecimalPipe, NgClass],
  template: `
    <div class="space-y-4 rounded-xl border border-dashed border-emerald-500/40 p-4">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 class="text-sm font-semibold">Valuta scambio</h3>
          <p class="text-xs opacity-60 mt-0.5">
            Confronta valore strutturale, forma recente e titolarità. Il verdetto è un
            supporto decisionale, non una garanzia di rendimento.
          </p>
        </div>
        <div class="flex items-center gap-2 text-xs">
          <span class="opacity-60">Modalità</span>
          <div class="inline-flex rounded-lg border overflow-hidden">
            <button
              type="button"
              class="px-3 py-1.5 font-medium transition-colors"
              [class.bg-emerald-600]="mode() === 'mantra'"
              [class.text-white]="mode() === 'mantra'"
              (click)="setMode('mantra')"
            >
              Mantra
            </button>
            <button
              type="button"
              class="px-3 py-1.5 font-medium transition-colors"
              [class.bg-emerald-600]="mode() === 'classic'"
              [class.text-white]="mode() === 'classic'"
              (click)="setMode('classic')"
            >
              Classic
            </button>
          </div>
        </div>
      </div>

      <!-- Two columns: Cedo / Ricevo -->
      <div class="grid gap-4 md:grid-cols-2">
        <section class="space-y-2">
          <h4 class="text-xs font-semibold uppercase tracking-wide opacity-70">
            Cedo ({{ giveIds().length }})
          </h4>
          <div class="max-h-56 overflow-y-auto space-y-1 rounded-lg border p-2">
            @for (p of ownOptions(); track p.playerId) {
              <label
                class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-black/5 dark:hover:bg-white/5"
                [class.bg-emerald-500/10]="isGive(p.playerId)"
              >
                <input
                  type="checkbox"
                  [checked]="isGive(p.playerId)"
                  (change)="toggleGive(p.playerId)"
                />
                <span class="flex-1 truncate">{{ p.name }}</span>
                <span class="text-[10px] opacity-50">{{ p.roles.join('/') }}</span>
                @if (p.fpCorr != null) {
                  <span class="tabular-nums text-xs opacity-70">{{ p.fpCorr | number: '1.0-0' }}</span>
                }
              </label>
            } @empty {
              <p class="text-xs opacity-50 px-2 py-3">Nessun giocatore in rosa.</p>
            }
          </div>
        </section>

        <section class="space-y-2">
          <h4 class="text-xs font-semibold uppercase tracking-wide opacity-70">
            Ricevo ({{ receiveIds().length }})
          </h4>
          <div class="max-h-56 overflow-y-auto space-y-1 rounded-lg border p-2">
            @for (p of marketOptions(); track p.playerId) {
              <label
                class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-black/5 dark:hover:bg-white/5"
                [class.bg-sky-500/10]="isReceive(p.playerId)"
              >
                <input
                  type="checkbox"
                  [checked]="isReceive(p.playerId)"
                  (change)="toggleReceive(p.playerId)"
                />
                <span class="flex-1 truncate">{{ p.name }}</span>
                <span class="text-[10px] opacity-50">{{ p.roles.join('/') }}</span>
                @if (p.fpCorr != null) {
                  <span class="tabular-nums text-xs opacity-70">{{ p.fpCorr | number: '1.0-0' }}</span>
                }
              </label>
            } @empty {
              <p class="text-xs opacity-50 px-2 py-3">
                Carica la dashboard scambi per vedere i target di mercato, oppure inserisci un ID sotto.
              </p>
            }
          </div>
          <div class="flex gap-2">
            <input
              type="text"
              class="flex-1 rounded-lg border bg-transparent px-3 py-1.5 text-sm"
              placeholder="Aggiungi playerId libero…"
              [(ngModel)]="freeReceiveId"
              (keydown.enter)="addFreeReceive(); $event.preventDefault()"
            />
            <button
              type="button"
              class="rounded-lg border px-3 py-1.5 text-xs font-medium"
              (click)="addFreeReceive()"
            >
              Aggiungi
            </button>
          </div>
        </section>
      </div>

      <div class="flex flex-wrap items-center gap-3">
        <label class="flex items-center gap-2 text-xs opacity-70">
          Tolleranza
          <input
            type="number"
            class="w-16 rounded border bg-transparent px-2 py-1 text-sm"
            min="0"
            max="50"
            step="1"
            [ngModel]="tolerance()"
            (ngModelChange)="tolerance.set(+$event || 0)"
          />
          %
        </label>
        <button
          type="button"
          class="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          [disabled]="evaluating() || !canEvaluate()"
          (click)="evaluate()"
        >
          {{ evaluating() ? 'Calcolo…' : 'Valuta equità' }}
        </button>
        @if (giveIds().length || receiveIds().length) {
          <button
            type="button"
            class="text-xs opacity-60 underline"
            (click)="clearSelection()"
          >
            Pulisci selezione
          </button>
        }
      </div>

      @if (error()) {
        <div class="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-600 dark:text-red-300">
          {{ error() }}
        </div>
      }

      @if (result(); as res) {
        <!-- Coverage warning above verdict (plan §7) -->
        @if (res.squadImpact?.warning) {
          <div class="rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm">
            ⚠ {{ res.squadImpact!.warning }}
          </div>
        }

        @if (res.seasonNotice) {
          <div class="rounded-lg border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-xs">
            {{ res.seasonNotice }}
          </div>
        }

        <!-- Verdict card -->
        <div
          class="rounded-xl border p-4 space-y-3"
          [ngClass]="verdictBorderClass(res)"
        >
          <div class="flex flex-wrap items-center gap-3">
            <span
              class="inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold"
              [ngClass]="verdictBadgeClass(res)"
            >
              {{ verdictLabel(res) }}
            </span>
            @if (res.valid && res.valueDeltaPercent != null) {
              <span class="tabular-nums text-sm font-medium">
                Δ {{ res.valueDeltaPercent > 0 ? '+' : '' }}{{ res.valueDeltaPercent | number: '1.1-1' }}%
              </span>
              <span class="text-xs opacity-50">
                (banda ±{{ res.toleranceBandPercent }}%)
              </span>
            }
            <span class="text-xs opacity-50 uppercase">{{ res.mode }}</span>
          </div>

          <!-- Delta bar -->
          @if (res.valid && res.valueDeltaPercent != null) {
            <div class="relative h-2 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
              <div
                class="absolute inset-y-0 left-1/2 w-px bg-black/30 dark:bg-white/30"
              ></div>
              <div
                class="absolute inset-y-0 rounded-full transition-all"
                [ngClass]="deltaBarClass(res.valueDeltaPercent)"
                [style.left.%]="deltaBarLeft(res.valueDeltaPercent)"
                [style.width.%]="deltaBarWidth(res.valueDeltaPercent)"
              ></div>
            </div>
          }

          @if (!res.valid && res.validationErrors.length) {
            <ul class="text-sm list-disc pl-5 space-y-0.5 text-red-600 dark:text-red-300">
              @for (e of res.validationErrors; track e) {
                <li>{{ e }}</li>
              }
            </ul>
          }

          <div class="grid gap-3 md:grid-cols-2">
            <div>
              <h5 class="text-xs font-semibold uppercase opacity-60 mb-1">Cedi</h5>
              <ul class="space-y-1">
                @for (p of res.give; track p.playerId) {
                  <li class="flex flex-wrap items-center gap-2 text-sm">
                    <span class="font-medium">{{ p.name }}</span>
                    <span class="tabular-nums opacity-70">PTV {{ p.ptv | number: '1.1-1' }}</span>
                    <span [ngClass]="confidenceClass(p.confidence)" class="text-[10px] uppercase rounded px-1.5 py-0.5">
                      {{ p.confidence }}
                    </span>
                    @for (f of p.flags; track f) {
                      <span class="text-[10px] rounded bg-amber-500/20 text-amber-700 dark:text-amber-300 px-1.5 py-0.5">
                        {{ f }}
                      </span>
                    }
                  </li>
                }
              </ul>
            </div>
            <div>
              <h5 class="text-xs font-semibold uppercase opacity-60 mb-1">Ricevi</h5>
              <ul class="space-y-1">
                @for (p of res.receive; track p.playerId) {
                  <li class="flex flex-wrap items-center gap-2 text-sm">
                    <span class="font-medium">{{ p.name }}</span>
                    <span class="tabular-nums opacity-70">PTV {{ p.ptv | number: '1.1-1' }}</span>
                    <span [ngClass]="confidenceClass(p.confidence)" class="text-[10px] uppercase rounded px-1.5 py-0.5">
                      {{ p.confidence }}
                    </span>
                    @for (f of p.flags; track f) {
                      <span class="text-[10px] rounded bg-amber-500/20 text-amber-700 dark:text-amber-300 px-1.5 py-0.5">
                        {{ f }}
                      </span>
                    }
                  </li>
                }
              </ul>
            </div>
          </div>

          @if (res.rationale.length) {
            <ul class="text-xs opacity-70 list-disc pl-4 space-y-0.5">
              @for (r of res.rationale; track r) {
                <li>{{ r }}</li>
              }
            </ul>
          }

          @if (res.squadImpact) {
            <div class="text-xs opacity-60 flex flex-wrap gap-2 pt-1">
              @for (entry of coverageEntries(res.squadImpact.coverageAfter); track entry[0]) {
                <span
                  class="rounded-full border px-2 py-0.5"
                  [class.border-emerald-500]="entry[1]"
                  [class.text-emerald-600]="entry[1]"
                  [class.border-red-400]="!entry[1]"
                  [class.text-red-500]="!entry[1]"
                >
                  {{ entry[0] }} {{ entry[1] ? '✓' : '✗' }}
                </span>
              }
            </div>
          }
        </div>
      }
    </div>
  `,
})
export class TradeEvaluatorComponent implements OnChanges {
  private readonly api = inject(MyTeamService);
  private readonly destroyRef = inject(DestroyRef);

  @Input({ required: true }) contextId!: string;
  @Input({ required: true }) sheetName!: string;
  @Input({ required: true }) teamName!: string;
  @Input() formationPrefs: string[] = ['4-3-3', '3-5-2', '3-4-3'];
  @Input() ruleset: 'MANTRA' | 'CLASSIC' = 'MANTRA';
  /** Optional dashboard data used to populate Cedo / Ricevo lists. */
  @Input() trades: TradesDashboardResponse | null = null;

  readonly mode = signal<TradeMode>('mantra');
  readonly giveIds = signal<string[]>([]);
  readonly receiveIds = signal<string[]>([]);
  readonly tolerance = signal(10);
  readonly evaluating = signal(false);
  readonly error = signal<string | null>(null);
  readonly result = signal<TradeEvaluateResponse | null>(null);

  readonly ownOptions = signal<EvaluatorPlayerOption[]>([]);
  readonly marketOptions = signal<EvaluatorPlayerOption[]>([]);

  freeReceiveId = '';

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['ruleset'] && this.ruleset) {
      this.mode.set(this.ruleset === 'CLASSIC' ? 'classic' : 'mantra');
    }
    if (changes['trades']) {
      this.rebuildOptions();
    }
  }

  private rebuildOptions(): void {
    const tr = this.trades;
    if (!tr) {
      this.ownOptions.set([]);
      this.marketOptions.set([]);
      return;
    }
    const own: EvaluatorPlayerOption[] = tr.tradeOutCandidates.map((c) => ({
      playerId: c.player.playerId,
      name: c.player.name,
      roles: c.player.roles ?? [],
      fpCorr: c.player.fpCorr,
      teamSerieA: c.player.teamSerieA,
      side: 'own',
    }));
    // Also surface excluded top performers as own-side options
    for (const e of tr.excludedTopPerformers ?? []) {
      if (!own.some((o) => o.playerId === e.player.playerId)) {
        own.push({
          playerId: e.player.playerId,
          name: e.player.name,
          roles: [],
          fpCorr: e.player.fpCorr,
          teamSerieA: e.player.teamSerieA,
          side: 'own',
        });
      }
    }
    const market: EvaluatorPlayerOption[] = tr.tradeInTargets.map((t) => ({
      playerId: t.playerId,
      name: t.name,
      roles: t.roles ?? [],
      fpCorr: t.fpCorr,
      side: 'market',
    }));
    this.ownOptions.set(own);
    this.marketOptions.set(market);
  }

  setMode(m: TradeMode): void {
    this.mode.set(m);
    this.result.set(null);
  }

  isGive(id: string): boolean {
    return this.giveIds().includes(id);
  }

  isReceive(id: string): boolean {
    return this.receiveIds().includes(id);
  }

  toggleGive(id: string): void {
    const cur = this.giveIds();
    this.giveIds.set(
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id],
    );
    this.result.set(null);
  }

  toggleReceive(id: string): void {
    const cur = this.receiveIds();
    this.receiveIds.set(
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id],
    );
    this.result.set(null);
  }

  addFreeReceive(): void {
    const id = this.freeReceiveId.trim();
    if (!id) return;
    if (!this.receiveIds().includes(id)) {
      this.receiveIds.set([...this.receiveIds(), id]);
      // surface in market list if missing
      if (!this.marketOptions().some((p) => p.playerId === id)) {
        this.marketOptions.set([
          ...this.marketOptions(),
          { playerId: id, name: `ID ${id}`, roles: [], side: 'market' },
        ]);
      }
    }
    this.freeReceiveId = '';
    this.result.set(null);
  }

  clearSelection(): void {
    this.giveIds.set([]);
    this.receiveIds.set([]);
    this.result.set(null);
    this.error.set(null);
  }

  canEvaluate(): boolean {
    return (
      !!this.contextId &&
      (this.giveIds().length > 0 || this.receiveIds().length > 0)
    );
  }

  evaluate(): void {
    if (!this.canEvaluate()) return;
    this.evaluating.set(true);
    this.error.set(null);
    this.api
      .evaluateTrade({
        contextId: this.contextId,
        sheetName: this.sheetName,
        teamName: this.teamName,
        mode: this.mode(),
        give: this.giveIds(),
        receive: this.receiveIds(),
        formationPrefs: this.formationPrefs,
        tolerancePercent: this.tolerance(),
      })
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.evaluating.set(false)),
      )
      .subscribe({
        next: (res) => this.result.set(res),
        error: (err) => {
          this.result.set(null);
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Valutazione fallita',
          );
        },
      });
  }

  // ── Presentation helpers ────────────────────────────────────────────────

  verdictLabel(res: TradeEvaluateResponse): string {
    if (!res.valid) return 'Non valido';
    switch (res.verdict) {
      case 'vantaggioso':
        return 'Vantaggioso';
      case 'sfavorevole':
        return 'Sfavorevole';
      case 'equilibrato':
        return 'Equilibrato';
      default:
        return '—';
    }
  }

  verdictBadgeClass(res: TradeEvaluateResponse): string {
    if (!res.valid) return 'bg-red-600 text-white';
    switch (res.verdict as TradeVerdict | null) {
      case 'vantaggioso':
        return 'bg-emerald-600 text-white';
      case 'sfavorevole':
        return 'bg-red-600 text-white';
      case 'equilibrato':
        return 'bg-amber-500 text-black';
      default:
        return 'bg-neutral-500 text-white';
    }
  }

  verdictBorderClass(res: TradeEvaluateResponse): string {
    if (!res.valid) return 'border-red-500/50';
    switch (res.verdict) {
      case 'vantaggioso':
        return 'border-emerald-500/50';
      case 'sfavorevole':
        return 'border-red-500/50';
      case 'equilibrato':
        return 'border-amber-500/50';
      default:
        return '';
    }
  }

  confidenceClass(c: TradeConfidence): string {
    switch (c) {
      case 'alta':
        return 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300';
      case 'media':
        return 'bg-sky-500/20 text-sky-700 dark:text-sky-300';
      case 'bassa':
        return 'bg-amber-500/20 text-amber-700 dark:text-amber-300';
      default:
        return 'bg-neutral-500/20 text-neutral-600 dark:text-neutral-300';
    }
  }

  deltaBarClass(delta: number): string {
    if (delta > 0) return 'bg-emerald-500';
    if (delta < 0) return 'bg-red-500';
    return 'bg-amber-400';
  }

  /** Map delta % into a centred bar (clamped ±25%). */
  deltaBarLeft(delta: number): number {
    const clamped = Math.max(-25, Math.min(25, delta));
    if (clamped >= 0) return 50;
    return 50 + (clamped / 25) * 50;
  }

  deltaBarWidth(delta: number): number {
    const clamped = Math.abs(Math.max(-25, Math.min(25, delta)));
    return (clamped / 25) * 50;
  }

  coverageEntries(m: Record<string, boolean>): [string, boolean][] {
    return Object.entries(m ?? {});
  }
}
