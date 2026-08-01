import { Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatePipe, DecimalPipe, PercentPipe } from '@angular/common';
import {
  NextSeasonPrediction,
  HybridPlayerPrediction,
  HybridStatsResponse,
  HybridStatus,
} from '../../core/models/api.models';
import { PredictionService } from '../../core/services/prediction.service';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { PredictionDrawerComponent } from './components/prediction-drawer/prediction-drawer.component';
import { PlayerDrawerComponent } from '../players/components/player-drawer/player-drawer.component';
import { PlayerSeasonStat } from '../../core/models/stats.models';

const SCORE_MIN = 4.5;
const SCORE_MAX = 9.0;

const MANTRA_ROLES = ['Por', 'Dc', 'Dd', 'Ds', 'B', 'E', 'M', 'C', 'T', 'W', 'A', 'Pc'];

/** User-friendly label definitions (Italian) */
const HYBRID_LABELS = [
  { id: 'ML_Confirmed', label: 'Confermato',          color: '#16a34a', desc: 'ML concorde col MANTRA, minutaggio garantito' },
  { id: 'ML_Risky',      label: 'Rischioso',           color: '#dc2626', desc: 'Prediction poco affidabile, confidence bassa' },
  { id: 'ML_Top',        label: 'Top',                 color: '#7c3aed', desc: 'Giocatore top riconosciuto dal ML' },
  { id: 'ML_Boosted',    label: 'Sorpresa',            color: '#a855f7', desc: 'ML molto sopra la media del ruolo, possibile sorpresa' },
  { id: 'Contradiction', label: 'Contrasto',           color: '#d97706', desc: 'Disaccordo MANTRA vs ML — valutare con cautela' },
  { id: 'Minutes_Risk',  label: 'Minuti a rischio',    color: '#f97316', desc: 'Pochi minuti previsti in stagione' },
  { id: 'Best_Value',    label: 'Miglior rapporto Q/P', color: '#22c55e', desc: 'Ottimo rapporto qualità/prezzo all\'asta' },
  { id: 'Sleeper',       label: 'Sleeper',             color: '#3b82f6', desc: 'Sottovalutato dal MANTRA ma con buona prediction ML' },
];

@Component({
  selector: 'app-predictions',
  standalone: true,
  imports: [
    FormsModule, DatePipe, DecimalPipe, PercentPipe, SkeletonComponent, ErrorBoundaryComponent,
    PredictionDrawerComponent, PlayerDrawerComponent,
  ],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Page header -->
      <div class="flex flex-col gap-2 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <div>
          <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Previsioni Giocatori</h1>
          <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">
            Punteggio ibrido MANTRA + ML per ogni giocatore
          </p>
        </div>
        <div class="flex rounded-lg border p-0.5 self-start sm:self-auto"
             style="border-color:var(--color-border);background:var(--color-surface)">
          @for (tab of tabs; track tab.id) {
            <button class="rounded-md px-3 py-1.5 text-xs font-medium transition-colors sm:px-4"
                    [style]="selectedTab() === tab.id
                      ? 'background:var(--color-accent);color:#fff'
                      : 'color:var(--color-text-secondary)'"
                    (click)="selectedTab.set(tab.id)">
              {{ tab.label }}
            </button>
          }
        </div>
      </div>

      <!-- ══════════════ TAB 1 — IBRIDO ════════════════════ -->
      @if (selectedTab() === 'hybrid') {
        <div class="p-4 sm:p-6">

          <!-- ── Stats bar ─────────────────────────────────── -->
          @if (hybridStats(); as s) {
            <div class="mb-5 grid grid-cols-2 gap-2 sm:grid-cols-4 sm:gap-3">
              <div class="rounded-lg border px-3 py-2.5"
                   style="border-color:var(--color-border);background:var(--color-surface)">
                <p class="text-[10px] font-medium uppercase tracking-wider" style="color:var(--color-text-secondary)"
                   [title]="FP_TOOLTIP">
                  Media FP Ibrido</p>
                <p class="text-lg font-bold tabular-nums mt-0.5" style="color:var(--color-accent)">
                  {{ s.avgFpIbrido | number:'1.1-1' }}</p>
              </div>
              <div class="rounded-lg border px-3 py-2.5"
                   style="border-color:var(--color-border);background:var(--color-surface)">
                <p class="text-[10px] font-medium uppercase tracking-wider" style="color:var(--color-text-secondary)"
                   [title]="CONF_TOOLTIP">
                  Media Confidence</p>
                <p class="text-lg font-bold tabular-nums mt-0.5" style="color:var(--color-accent)">
                  {{ s.avgConfidenceScore | number:'1.1-1' }}</p>
              </div>
              <div class="rounded-lg border px-3 py-2.5"
                   style="border-color:var(--color-border);background:var(--color-surface)">
                <p class="text-[10px] font-medium uppercase tracking-wider" style="color:var(--color-text-secondary)"
                   [title]="GAP_TOOLTIP">
                  Gap Medio MANTRA–ML</p>
                <p class="text-lg font-bold tabular-nums mt-0.5"
                    [style]="'color:' + (s.avgFpGap >= 0 ? '#16a34a' : '#ea580c')">
                  {{ s.avgFpGap >= 0 ? '+' : '' }}{{ s.avgFpGap | number:'1.1-1' }}</p>
              </div>
              <div class="rounded-lg border px-3 py-2.5"
                   style="border-color:var(--color-border);background:var(--color-surface)">
                <p class="text-[10px] font-medium uppercase tracking-wider" style="color:var(--color-text-secondary)">
                  Con dati ML</p>
                <p class="text-lg font-bold tabular-nums mt-0.5" style="color:var(--color-accent)">
                  {{ s.pctWithMl | percent:'1.0-0' }}</p>
              </div>
              @if (lastGenerated(); as ts) {
                <div class="rounded-lg border px-3 py-2.5 col-span-2 sm:col-span-4"
                     style="border-color:var(--color-border);background:var(--color-surface)">
                  <p class="text-[10px]" style="color:var(--color-text-secondary)">
                    Ultimo aggiornamento: {{ ts | date:'d MMM y, HH:mm' }}
                  </p>
                </div>
              }
            </div>
          }

          <!-- ── Readiness banner ──────────────────────────── -->
          @if (!statusLoading() && readinessMessage(); as msg) {
            <div class="mb-4 rounded-lg border px-4 py-3 text-sm"
                 style="background:#fef3c7;border-color:#f59e0b;color:#92400e">
              <p class="font-medium mb-1">⚙️ Calcoli non ancora eseguiti</p>
              <ul class="list-disc list-inside space-y-0.5 text-xs">
                @for (m of msg; track m) { <li>{{ m }}</li> }
              </ul>
              <p class="text-xs mt-1">
                Vai su <strong>Admin → Pipeline</strong> per verificare e generare i dati necessari.
              </p>
            </div>
          }

          <!-- ── Filter bar ────────────────────────────────── -->
          <div class="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
            <div class="relative flex-1 max-w-xs">
              <svg class="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   width="14" height="14" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2"
                   style="color:var(--color-text-secondary)">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
              </svg>
              <input class="w-full rounded-lg border py-1.5 pl-8 pr-3 text-sm outline-none"
                     style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                     placeholder="Cerca giocatore…"
                     [ngModel]="hybridSearch()"
                     (ngModelChange)="hybridSearch.set($event)" />
            </div>

            <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                    style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                    [ngModel]="hybridRuolo()"
                    (ngModelChange)="hybridRuolo.set($event)">
              <option value="">Tutti i ruoli</option>
              @for (r of MANTRA_ROLES; track r) {
                <option [value]="r">{{ r }}</option>
              }
            </select>

            <input class="w-20 rounded-lg border px-2 py-1.5 text-sm outline-none"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   type="number" min="0" max="100" placeholder="FP min"
                   [title]="FP_TOOLTIP"
                   [ngModel]="fpMin()"
                   (ngModelChange)="fpMin.set($event)" />
            <input class="w-20 rounded-lg border px-2 py-1.5 text-sm outline-none"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   type="number" min="0" max="100" placeholder="FP max"
                   [title]="FP_TOOLTIP"
                   [ngModel]="fpMax()"
                   (ngModelChange)="fpMax.set($event)" />

            <div class="flex flex-wrap gap-1">
              @for (preset of confidencePresets; track preset.value) {
                <button class="rounded-full border px-2.5 py-1 text-xs font-medium"
                        [style]="hybridConfidenceMin() === preset.value
                          ? 'background:var(--color-accent);color:#fff;border-color:transparent'
                          : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                        (click)="hybridConfidenceMin.set(preset.value)">
                  {{ preset.label }}
                </button>
              }
            </div>

            <span class="text-xs whitespace-nowrap" style="color:var(--color-text-secondary)">
              {{ filterTotal() }} giocatori
            </span>
          </div>

          <!-- ── Label filter pills ────────────────────────── -->
          <div class="mb-5 flex flex-wrap gap-1.5">
            <span class="text-xs font-medium py-1" style="color:var(--color-text-secondary)">Filtra per etichetta:</span>
            @for (l of HYBRID_LABELS; track l.id) {
              <button class="rounded-full border px-2.5 py-0.5 text-xs font-medium transition-all"
                      [style]="activeLabels().has(l.id)
                        ? 'background:' + l.color + ';color:#fff;border-color:transparent'
                        : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border);opacity:0.7'"
                      [title]="l.desc"
                      (click)="toggleLabel(l.id)">
                {{ l.label }}
              </button>
            }
            @if (activeLabels().size > 0) {
              <button class="text-xs underline py-0.5" style="color:var(--color-text-secondary)"
                      (click)="clearLabels()">
                Cancella filtri
              </button>
            }
          </div>

          <!-- ── Loading / Error / Player cards ────────────── -->
          @if (hybridLoading() && !hybridError()) {
            <div class="space-y-2">
              @for (_ of skeletonRows; track $index) { <app-skeleton height="64px" /> }
            </div>
          } @else if (hybridError()) {
            <app-error-boundary [message]="hybridError()!" />
          } @else if (filteredHybrid().length === 0 && !readinessMessage()) {
            <div class="rounded-lg border px-6 py-12 text-center text-sm"
                 style="border-color:var(--color-border);color:var(--color-text-secondary)">
              Nessuna previsione disponibile.
            </div>
          } @else if (filteredHybrid().length > 0) {
            <div class="space-y-1.5">
              @for (p of paginatedHybrid(); track p.playerName; let i = $index) {
                <!-- Player card -->
                <div class="rounded-lg border px-3 py-2.5 cursor-pointer transition-all hover:opacity-85 hover:border-opacity-60 sm:px-4 sm:py-3"
                     style="border-color:var(--color-border);background:var(--color-surface)"
                     (click)="selectedPlayer.set(p)">
                  <div class="flex items-center gap-3">
                    <!-- Rank -->
                    <span class="w-6 text-right text-xs font-mono shrink-0" style="color:var(--color-text-secondary)">
                      {{ (hybridPage() - 1) * hybridPageSize() + i + 1 }}
                    </span>

                    <!-- Player info -->
                    <div class="min-w-0 flex-1">
                      <div class="flex items-center gap-2">
                        <p class="font-medium text-sm truncate" style="color:var(--color-text-primary)">
                          {{ p.playerName ?? '—' }}
                        </p>
                        @if (p.team) {
                          <span class="text-[10px] px-1.5 py-0.5 rounded shrink-0"
                                style="background:var(--color-surface-raised);color:var(--color-text-secondary)">
                            {{ p.team }}
                          </span>
                        }
                      </div>
                      <div class="flex items-center gap-2 mt-0.5">
                        <span class="text-[11px] font-mono font-medium rounded px-1.5 py-0.5"
                              style="background:var(--color-brand-700);color:var(--color-brand-100)">
                          {{ p.ruoloPrimario ?? '—' }}
                        </span>
                        @for (l of p.hybridLabels ?? []; track l) {
                          @if (labelColor(l); as c) {
                            <span class="inline-block rounded-full px-1.5 py-0.5 text-[10px] font-medium whitespace-nowrap"
                                  [style]="'background:' + c + '22;color:' + c + ';border:1px solid ' + c + '44'">
                              {{ userLabel(l) }}
                            </span>
                          }
                        }
                        @if (!p.hasMlData && (p.hybridLabels ?? []).length === 0) {
                          <span class="text-[10px]" style="color:var(--color-text-secondary)">Solo MANTRA</span>
                        }
                      </div>
                    </div>

                    <!-- Score column -->
                    <div class="flex items-center gap-4 shrink-0">
                      <!-- FP Ibrido big -->
                      <div class="text-right min-w-[60px]">
                        <p class="text-xs text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)"
                           [title]="FP_TOOLTIP">
                          FP Ibrido</p>
                        <p class="text-lg font-bold tabular-nums leading-tight" style="color:var(--color-accent)">
                          {{ p.fpIbrido != null ? (p.fpIbrido | number:'1.1-1') : '—' }}
                        </p>
                        <!-- mini bar -->
                        <div class="w-full h-1 rounded-full mt-0.5 overflow-hidden"
                             style="background:var(--color-surface-raised)">
                          <div class="h-full rounded-full"
                               [style]="'background:var(--color-accent);width:' + (p.fpIbrido ?? 0) + '%'"></div>
                        </div>
                      </div>

                      <!-- Confidence + Gap mini -->
                      <div class="text-right min-w-[44px]">
                        <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)"
                           [title]="CONF_TOOLTIP">
                          Conf</p>
                        @if (p.confidenceScore != null) {
                          <p class="text-sm font-semibold tabular-nums"
                             [style]="'color:' + (p.confidenceScore >= 70 ? '#16a34a' : p.confidenceScore >= 50 ? '#d97706' : '#dc2626')">
                            {{ p.confidenceScore | number:'1.0-0' }}
                          </p>
                        } @else { <p class="text-sm" style="color:var(--color-text-secondary)">—</p> }
                      </div>

                      <div class="text-right min-w-[44px]">
                        <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)"
                           [title]="GAP_TOOLTIP">
                          Gap</p>
                        @if (p.fpGap != null) {
                          <p class="text-sm font-semibold tabular-nums"
                             [style]="'color:' + (p.fpGap >= 0 ? '#16a34a' : '#ea580c')">
                            {{ p.fpGap >= 0 ? '+' : '' }}{{ p.fpGap | number:'1.0-0' }}
                          </p>
                        } @else { <p class="text-sm" style="color:var(--color-text-secondary)">—</p> }
                      </div>

                      <!-- Chevron -->
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                           stroke="currentColor" stroke-width="2" style="color:var(--color-text-secondary)">
                        <path d="m9 18 6-6-6-6"/>
                      </svg>
                    </div>
                  </div>
                </div>
              }
            </div>

            <!-- Pagination -->
            <div class="mt-4 flex items-center justify-between text-xs"
                 style="color:var(--color-text-secondary)">
              <span>{{ totalDisplayed() }} / {{ filterTotal() }} giocatori</span>
              <div class="flex gap-2 items-center">
                <button class="rounded border px-3 py-1.5 disabled:opacity-40 font-medium"
                        [disabled]="hybridPage() <= 1"
                        (click)="hybridPage.set(hybridPage() - 1)"
                        style="border-color:var(--color-border)">← Precedenti</button>
                <span class="px-2 py-1 font-mono">{{ hybridPage() }} / {{ totalPages() }}</span>
                <button class="rounded border px-3 py-1.5 disabled:opacity-40 font-medium"
                        [disabled]="hybridPage() >= totalPages()"
                        (click)="hybridPage.set(hybridPage() + 1)"
                        style="border-color:var(--color-border)">Successivi →</button>
              </div>
            </div>
          }
        </div>
      }

      <!-- ══════════════ TAB 2 — NEXT SEASON ═════════════════ -->
      @if (selectedTab() === 'next') {
        <div class="p-4 sm:p-6">
          <div class="relative mb-4 max-w-xs">
            <svg class="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                 width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2"
                 style="color:var(--color-text-secondary)">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
            </svg>
            <input class="w-full rounded-lg border py-1.5 pl-8 pr-3 text-sm outline-none"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   placeholder="Cerca giocatore…"
                   [ngModel]="nextSearch()"
                   (ngModelChange)="nextSearch.set($event)" />
          </div>

          @if (nextLoading()) {
            <div class="space-y-2">
              @for (_ of skeletonRows; track $index) { <app-skeleton height="48px" /> }
            </div>
          } @else if (nextMlUnavailable()) {
            <app-error-boundary title="Pipeline Non Pronta"
              message="I dati ML per la prossima stagione sono in elaborazione. Riprova tra qualche minuto." />
          } @else if (nextError()) {
            <app-error-boundary [message]="nextError()!" />
          } @else {
            <div class="rounded-lg border overflow-hidden" style="border-color:var(--color-border)">
              <div class="space-y-px">
                @for (p of filteredNext(); let i = $index; track p.playerName) {
                  <div class="flex items-center gap-3 px-4 py-2.5 transition-colors"
                       [class.cursor-pointer]="p.playerFotmobId != null"
                       [title]="p.playerFotmobId == null ? 'Nessun id giocatore disponibile per il dettaglio' : ''"
                       style="background:var(--color-surface)"
                       (click)="p.playerFotmobId != null && nextSelectedPlayer.set(toPlayerSeasonStat(p))">
                    <span class="w-7 text-right font-mono text-xs shrink-0"
                          style="color:var(--color-text-secondary)">{{ i + 1 }}</span>
                    <p class="min-w-0 flex-1 truncate text-sm font-medium"
                       style="color:var(--color-text-primary)">{{ p.playerName }}</p>
                    <div class="shrink-0 flex items-center gap-3">
                      <div class="w-20 h-1.5 rounded-full overflow-hidden"
                           style="background:var(--color-surface-raised)">
                        <div class="h-full rounded-full" style="background:var(--color-accent)"
                             [style.width.%]="scorePct(p.predictedNextFantavoto)"></div>
                      </div>
                      <span class="font-bold text-sm tabular-nums" style="color:var(--color-accent)">
                        {{ p.predictedNextFantavoto | number:'1.2-2' }}
                      </span>
                    </div>
                  </div>
                }
              </div>
            </div>
          }
        </div>
      }
    </div>

    @if (selectedPlayer(); as p) {
      <app-prediction-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
    @if (nextSelectedPlayer(); as p) {
      <app-player-drawer [player]="p" (closed)="nextSelectedPlayer.set(null)" />
    }
  `,
})
export class PredictionsComponent {
  private readonly predService = inject(PredictionService);

  readonly FP_TOOLTIP = 'Media pesata tra il punteggio storico MANTRA (FP_Corr) e la predizione ML normalizzata, 50/50 di default — scala 0-100.';
  readonly CONF_TOOLTIP = 'Affidabilità della stima (0-100): combina la certezza del modello ML (bassa deviazione standard) e i minuti attesi in campo. 0 = nessun dato ML disponibile.';
  readonly GAP_TOOLTIP = 'Differenza tra il punteggio MANTRA storico e la predizione ML (entrambi su scala 0-100). Positivo = MANTRA più ottimista di ML; negativo = il contrario. Oltre ±30 punti scatta l\'etichetta \'Contrasto\'.';

  readonly tabs = [
    { id: 'hybrid' as const, label: 'Ibrido' },
    { id: 'next' as const, label: 'Next Season' },
  ];
  readonly skeletonRows = Array.from({ length: 8 });
  readonly selectedTab = signal<'hybrid' | 'next'>('hybrid');

  // ── Hybrid tab state ──────────────────────────────────
  readonly hybridItems = signal<HybridPlayerPrediction[]>([]);
  readonly hybridStats = signal<HybridStatsResponse | null>(null);
  readonly hybridLoading = signal(true);
  readonly hybridError = signal<string | null>(null);
  readonly hybridSearch = signal('');
  readonly hybridRuolo = signal('');
  readonly hybridConfidenceMin = signal<number | null>(null);
  readonly fpMin = signal<number | null>(null);
  readonly fpMax = signal<number | null>(null);
  readonly hybridPage = signal(1);
  readonly hybridPageSize = signal(50);
  readonly sortField = signal<string | null>('fpIbrido');
  readonly sortDir = signal<'asc' | 'desc'>('desc');
  readonly activeLabels = signal(new Set<string>());
  readonly selectedPlayer = signal<HybridPlayerPrediction | null>(null);
  readonly lastGenerated = signal<string | null>(null);

  // ── Readiness status ─────────────────────────────────
  readonly hybridStatus = signal<HybridStatus | null>(null);
  readonly statusLoading = signal(true);

  readonly confidencePresets = [
    { label: 'Tutti', value: null as number | null },
    { label: '≥70', value: 70 },
    { label: '≥50', value: 50 },
    { label: '<30', value: -1 },
  ];

  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.filterTotal() / this.hybridPageSize())));
  readonly filterTotal = computed(() => this.applyFilters(this.hybridItems()).length);

  readonly totalDisplayed = computed(() => this.paginatedHybrid().length);

  private applyFilters(items: HybridPlayerPrediction[]): HybridPlayerPrediction[] {
    const q = this.hybridSearch().toLowerCase();
    const ruolo = this.hybridRuolo();
    const confMin = this.hybridConfidenceMin();
    const labels = this.activeLabels();
    const fpMin = this.fpMin();
    const fpMax = this.fpMax();

    if (q) items = items.filter(p => (p.playerName ?? '').toLowerCase().includes(q));
    if (ruolo) items = items.filter(p => p.ruoloPrimario === ruolo);
    if (confMin !== null) {
      if (confMin === -1) items = items.filter(p => (p.confidenceScore ?? 100) < 30);
      else items = items.filter(p => (p.confidenceScore ?? 0) >= confMin);
    }
    if (fpMin !== null) items = items.filter(p => (p.fpIbrido ?? -1) >= fpMin);
    if (fpMax !== null) items = items.filter(p => (p.fpIbrido ?? Infinity) <= fpMax);
    if (labels.size > 0) {
      items = items.filter(p => (p.hybridLabels ?? []).some(l => labels.has(l)));
    }
    return items;
  }

  private applySort(items: HybridPlayerPrediction[]): HybridPlayerPrediction[] {
    const sf = this.sortField();
    const sd = this.sortDir();
    if (!sf) return items;
    return [...items].sort((a, b) => {
      const av = (a as any)[sf] ?? -999999;
      const bv = (b as any)[sf] ?? -999999;
      return sd === 'desc' ? bv - av : av - bv;
    });
  }

  readonly filteredHybrid = computed(() => {
    return this.applySort(this.applyFilters(this.hybridItems()));
  });

  readonly paginatedHybrid = computed(() => {
    const all = this.filteredHybrid();
    const page = this.hybridPage();
    const size = this.hybridPageSize();
    return all.slice((page - 1) * size, page * size);
  });

  // ── Next season tab state ─────────────────────────────
  readonly nextItems = signal<NextSeasonPrediction[]>([]);
  readonly nextLoaded = signal(false);
  readonly nextLoading = signal(false);
  readonly nextError = signal<string | null>(null);
  readonly nextMlUnavailable = signal(false);
  readonly nextSearch = signal('');
  /** Next Season row opened in the (reused) PlayerDrawerComponent. */
  readonly nextSelectedPlayer = signal<PlayerSeasonStat | null>(null);

  /**
   * Adapts a NextSeasonPrediction row (playerName/playerFotmobId/
   * predictedNextFantavoto only) into the shape PlayerDrawerComponent
   * needs. Only the 4 fields the drawer actually reads (player_fotmob_id,
   * player_name, team_name, stat_category) are meaningful here — the rest
   * of PlayerSeasonStat has no equivalent in NextSeasonPrediction, hence
   * the type assertion rather than a fully-populated fake object.
   */
  toPlayerSeasonStat(p: NextSeasonPrediction): PlayerSeasonStat {
    // Only invoked from the template when p.playerFotmobId != null (see the
    // click guard on the Next Season row), hence the non-null assertion.
    return {
      player_fotmob_id: p.playerFotmobId!,
      player_name: p.playerName,
      team_name: null,
      stat_category: '',
    } as PlayerSeasonStat;
  }

  readonly filteredNext = computed(() => {
    const q = this.nextSearch().toLowerCase();
    return this.nextItems()
      .filter(p => !q || (p.playerName ?? '').toLowerCase().includes(q))
      .sort((a, b) => b.predictedNextFantavoto - a.predictedNextFantavoto);
  });

  // ── Constants ─────────────────────────────────────────
  readonly MANTRA_ROLES = MANTRA_ROLES;
  readonly HYBRID_LABELS = HYBRID_LABELS;

  // ── Derived status messages ───────────────────────────
  readonly readinessMessage = computed<string[] | null>(() => {
    const status = this.hybridStatus();
    if (!status) return null;
    const missing: string[] = [];
    if (!status.mlPredictionsReady) missing.push('calcolo ML predictions (risultati del modello)');
    if (status.mantraResults.length === 0) missing.push('calcolo MANTRA (voti storici)');
    if (!status.hybridReady) {
      if (status.mantraResults.length > 0 && status.mlPredictionsReady) {
        missing.push('calcolo ibrido MANTRA+ML (esegui "Salva e Rigenera" nella sezione Admin → Pipeline)');
      }
    }
    return missing.length > 0 ? missing : null;
  });

  // ── Methods ───────────────────────────────────────────

  scorePct(v: number): number {
    return Math.max(0, Math.min(100, ((v - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100));
  }

  labelColor(label: string): string | null {
    const found = HYBRID_LABELS.find(l => l.id === label);
    return found ? found.color : null;
  }

  userLabel(label: string): string {
    const found = HYBRID_LABELS.find(l => l.id === label);
    return found ? found.label : label;
  }

  toggleLabel(id: string) {
    this.activeLabels.update(s => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  clearLabels() {
    this.activeLabels.set(new Set());
  }

  private loadStatus() {
    this.statusLoading.set(true);
    this.predService.getHybridStatus().subscribe({
      next: s => { this.hybridStatus.set(s); this.statusLoading.set(false); },
      error: () => { this.hybridStatus.set(null); this.statusLoading.set(false); },
    });
  }

  private loadHybridData() {
    this.hybridLoading.set(true);
    this.hybridError.set(null);
    this.predService.getHybridPredictions({ page: 1, size: 2000 }).subscribe({
      next: res => {
        this.hybridItems.set(res.items);
        this.hybridLoading.set(false);
        const ts = res.meta?.generatedAt;
        if (ts) this.lastGenerated.set(ts);
      },
      error: err => {
        this.hybridError.set('Calcolo ibrido non disponibile. Verifica che MANTRA e ML predictions siano stati generati.');
        this.hybridLoading.set(false);
      },
    });
    this.predService.getHybridStats().subscribe({
      next: s => this.hybridStats.set(s),
      error: () => {},
    });
  }

  private lastHybridFilterSignature = '';

  constructor() {
    this.loadStatus();
    this.loadHybridData();

    // Reset to page 1 whenever a filter (not the page itself) changes —
    // no HTTP call needed here since filteredHybrid/paginatedHybrid are
    // plain computed signals over the already-loaded hybridItems.
    effect(() => {
      const sig = JSON.stringify([
        this.hybridSearch(), this.hybridRuolo(), this.hybridConfidenceMin(),
        [...this.activeLabels()], this.fpMin(), this.fpMax(),
      ]);
      if (sig !== this.lastHybridFilterSignature) {
        this.lastHybridFilterSignature = sig;
        this.hybridPage.set(1);
      }
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
          else this.nextError.set('Impossibile caricare le previsioni Next Season.');
          this.nextLoading.set(false);
        },
      });
    });
  }
}
