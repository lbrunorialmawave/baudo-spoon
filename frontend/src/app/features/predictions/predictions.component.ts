import { Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, PercentPipe } from '@angular/common';
import {
  NextSeasonPrediction,
  ModelComparison,
  HybridPlayerPrediction,
  HybridStatsResponse,
  HybridConfig,
} from '../../core/models/api.models';
import { PredictionService } from '../../core/services/prediction.service';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

const SCORE_MIN = 4.5;
const SCORE_MAX = 9.0;

const MANTRA_ROLES = ['Por', 'Dc', 'Dd', 'Ds', 'B', 'E', 'M', 'C', 'T', 'W', 'A', 'Pc'];

const HYBRID_LABELS = [
  { id: 'ML_Confirmed', label: 'ML Confirmed', color: '#16a34a' },
  { id: 'ML_Risky', label: 'ML Risky', color: '#dc2626' },
  { id: 'ML_Boosted', label: 'ML Boosted', color: '#7c3aed' },
  { id: 'Contradiction', label: 'Contradiction', color: '#d97706' },
  { id: 'Minutes_Risk', label: 'Minutes Risk', color: '#f97316' },
  { id: 'Best_Value', label: 'Best Value', color: '#22c55e' },
  { id: 'Sleeper', label: 'Sleeper', color: '#3b82f6' },
];

@Component({
  selector: 'app-predictions',
  standalone: true,
  imports: [FormsModule, DecimalPipe, PercentPipe, SkeletonComponent, ErrorBoundaryComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Page header -->
      <div class="flex flex-col gap-2 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Predictions</h1>
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

      <!-- ══════════════ TAB 1 — HYBRID ════════════════════ -->
      @if (selectedTab() === 'hybrid') {
        <div class="p-4 sm:p-6">
          <!-- Stats strip -->
          @if (hybridStats(); as s) {
            <div class="mb-4 flex flex-wrap gap-3 text-xs"
                 style="color:var(--color-text-secondary)">
              <span class="rounded-lg border px-3 py-1.5"
                    style="border-color:var(--color-border);background:var(--color-surface)">
                MANTRA <strong>{{ (config()?.PESO_MANTRA ?? 0.5) | percent }}</strong>
                / ML <strong>{{ (config()?.PESO_ML ?? 0.5) | percent }}</strong>
              </span>
              <span class="rounded-lg border px-3 py-1.5"
                    style="border-color:var(--color-border);background:var(--color-surface)">
                avg FP Ibrido: <strong>{{ s.avgFpIbrido | number:'1.1-1' }}</strong>
              </span>
              <span class="rounded-lg border px-3 py-1.5"
                    style="border-color:var(--color-border);background:var(--color-surface)">
                avg Confidence: <strong>{{ s.avgConfidenceScore | number:'1.1-1' }}</strong>
              </span>
              <span class="rounded-lg border px-3 py-1.5"
                    style="border-color:var(--color-border);background:var(--color-surface)">
                avg Gap: <strong>{{ s.avgFpGap | number:'1.1-1' }}</strong>
              </span>
              <span class="rounded-lg border px-3 py-1.5"
                    style="border-color:var(--color-border);background:var(--color-surface)">
                {{ s.pctWithMl | percent }} with ML data
              </span>
              @if (lastGenerated(); as ts) {
                <span class="rounded-lg border px-3 py-1.5"
                      style="border-color:var(--color-border);background:var(--color-surface)">
                  Updated: {{ ts | date:'d MMM HH:mm' }}
                </span>
              }
            </div>
          }

          <!-- Filters -->
          <div class="mb-4 flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3">
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-55"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   placeholder="Search player…"
                   [ngModel]="hybridSearch()"
                   (ngModelChange)="hybridSearch.set($event)" />

            <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                    style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                    [ngModel]="hybridRuolo()"
                    (ngModelChange)="hybridRuolo.set($event)">
              <option value="">All roles</option>
              @for (r of MANTRA_ROLES; track r) {
                <option [value]="r">{{ r }}</option>
              }
            </select>

            <!-- Confidence quick filters -->
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
          </div>

          <!-- Hybrid label pills -->
          <div class="mb-4 flex flex-wrap gap-1.5">
            @for (l of HYBRID_LABELS; track l.id) {
              <button class="rounded-full border px-2.5 py-0.5 text-xs font-medium transition-opacity"
                      [style]="activeLabels().has(l.id)
                        ? 'background:' + l.color + ';color:#fff;border-color:transparent'
                        : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border);opacity:0.7'"
                      (click)="toggleLabel(l.id)">
                {{ l.label }}
              </button>
            }
          </div>

          <!-- Loading / Error / Table -->
          @if (hybridLoading()) {
            <div class="space-y-2">
              @for (_ of skeletonRows; track $index) { <app-skeleton height="52px" /> }
            </div>
          } @else if (hybridError()) {
            <app-error-boundary [message]="hybridError()!" />
          } @else if (filteredHybrid().length === 0) {
            <div class="rounded-lg border px-6 py-12 text-center text-sm"
                 style="border-color:var(--color-border);color:var(--color-text-secondary)">
              No hybrid predictions available.
            </div>
          } @else {
            <div class="overflow-x-auto rounded-lg border"
                 style="border-color:var(--color-border)">
              <table class="w-full text-sm">
                <thead>
                  <tr style="background:var(--color-surface)">
                    <th class="w-8 px-3 py-2 text-left font-medium text-xs"
                        style="color:var(--color-text-secondary)">#</th>
                    <th class="px-3 py-2 text-left font-medium text-xs cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('playerName')">Player</th>
                    <th class="px-3 py-2 text-left font-medium text-xs hidden sm:table-cell"
                        style="color:var(--color-text-secondary)">Ruolo</th>
                    <th class="px-3 py-2 text-right font-medium text-xs cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('FP_Corr')">FP Corr</th>
                    <th class="px-3 py-2 text-right font-medium text-xs cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('predictedFantavoto')">Predicted</th>
                    <th class="px-3 py-2 text-right font-medium text-xs cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('fpIbrido')">FP Ibrido</th>
                    <th class="px-3 py-2 text-right font-medium text-xs hidden md:table-cell cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('confidenceScore')">Conf</th>
                    <th class="px-3 py-2 text-right font-medium text-xs hidden lg:table-cell cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('expectedValue')">Pts Stag</th>
                    <th class="px-3 py-2 text-right font-medium text-xs hidden lg:table-cell cursor-pointer"
                        style="color:var(--color-text-secondary)"
                        (click)="setSort('fpGap')">Gap</th>
                    <th class="px-3 py-2 text-left font-medium text-xs"
                        style="color:var(--color-text-secondary)">Labels</th>
                  </tr>
                </thead>
                <tbody class="divide-y" style="border-color:var(--color-border)">
                  @for (p of filteredHybrid(); track p.playerName; let i = $index) {
                    <tr class="cursor-pointer hover:opacity-80"
                        style="border-color:var(--color-border)"
                        (click)="selectedPlayer.set(p)">
                      <td class="w-8 px-3 py-2 text-xs font-mono"
                          style="color:var(--color-text-secondary)">{{ i + 1 }}</td>
                      <td class="px-3 py-2">
                        <p class="font-medium truncate max-w-36 sm:max-w-52"
                           style="color:var(--color-text-primary)">{{ p.playerName }}</p>
                        <p class="text-xs truncate" style="color:var(--color-text-secondary)">{{ p.team }}</p>
                      </td>
                      <td class="px-3 py-2 hidden sm:table-cell"
                          style="color:var(--color-text-secondary)">{{ p.ruoloPrimario }}</td>
                      <td class="px-3 py-2 text-right font-mono text-xs"
                          style="color:var(--color-text-secondary)">{{ (p.FP_Corr ?? '—') | number:'1.1-1' }}</td>
                      <td class="px-3 py-2 text-right font-mono text-xs"
                          style="color:var(--color-accent);font-weight:600">
                        {{ (p.predictedFantavoto ?? '—') | number:'1.2-2' }}
                      </td>
                      <td class="px-3 py-2 text-right">
                        <div class="flex items-center justify-end gap-1.5">
                          <div class="w-14 h-1.5 rounded-full overflow-hidden"
                               style="background:var(--color-surface-raised)">
                            <div class="h-full rounded-full"
                                 [style]="'background:var(--color-accent);width:' + (p.fpIbrido ?? 0) + '%'"></div>
                          </div>
                          <span class="font-mono text-xs font-semibold" style="color:var(--color-accent)">
                            {{ (p.fpIbrido ?? '—') | number:'1.1-1' }}
                          </span>
                        </div>
                      </td>
                      <td class="px-3 py-2 text-right hidden md:table-cell">
                        @if (p.confidenceScore != null) {
                          <div class="flex items-center justify-end gap-1.5">
                            <div class="w-12 h-1.5 rounded-full overflow-hidden"
                                 style="background:var(--color-surface-raised)">
                              <div class="h-full rounded-full"
                                   [style]="'width:' + p.confidenceScore + '%;background:'
                                   + (p.confidenceScore >= 70 ? '#16a34a' : p.confidenceScore >= 40 ? '#d97706' : '#dc2626')">
                              </div>
                            </div>
                            <span class="font-mono text-xs"
                                  [style]="'color:' + (p.confidenceScore >= 70 ? '#16a34a' : p.confidenceScore >= 40 ? '#d97706' : '#dc2626')">
                              {{ p.confidenceScore | number:'1.0-0' }}
                            </span>
                          </div>
                        } @else { <span style="color:var(--color-text-secondary)">—</span> }
                      </td>
                      <td class="px-3 py-2 text-right font-mono text-xs hidden lg:table-cell"
                          style="color:var(--color-text-secondary)">
                        {{ p.expectedValue != null ? (p.expectedValue | number:'1.0-0') : '—' }}
                      </td>
                      <td class="px-3 py-2 text-right hidden lg:table-cell">
                        @if (p.fpGap != null) {
                          <span class="inline-flex items-center gap-0.5 font-mono text-xs"
                                [style]="'color:' + (p.fpGap >= 0 ? '#16a34a' : '#ea580c')">
                            @if (p.fpGap >= 0) {
                              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"><path d="M5 0l5 8H0z"/></svg>
                            } @else {
                              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"><path d="M0 2h10L5 10z"/></svg>
                            }
                            {{ p.fpGap | number:'1.1-1' }}
                          </span>
                        } @else { <span style="color:var(--color-text-secondary)">—</span> }
                      </td>
                      <td class="px-3 py-2">
                        <div class="flex flex-wrap gap-1">
                          @for (l of p.hybridLabels; track l) {
                            @if (labelColor(l); as c) {
                              <span class="inline-block rounded-full px-1.5 py-0.5 text-[10px] font-medium"
                                    [style]="'background:' + c + '22;color:' + c + ';border:1px solid ' + c + '44'">
                                {{ l }}
                              </span>
                            }
                          }
                        </div>
                      </td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>

            <!-- Pagination -->
            <div class="mt-3 flex items-center justify-between text-xs"
                 style="color:var(--color-text-secondary)">
              <span>{{ filteredHybrid().length }} players</span>
              <div class="flex gap-2">
                <button class="rounded border px-3 py-1 disabled:opacity-40"
                        [disabled]="hybridPage() <= 1"
                        (click)="hybridPage.set(hybridPage() - 1)"
                        style="border-color:var(--color-border)">Prev</button>
                <span class="px-2 py-1">{{ hybridPage() }} / {{ totalPages() }}</span>
                <button class="rounded border px-3 py-1 disabled:opacity-40"
                        [disabled]="hybridPage() >= totalPages()"
                        (click)="hybridPage.set(hybridPage() + 1)"
                        style="border-color:var(--color-border)">Next</button>
              </div>
            </div>
          }
        </div>
      }

      <!-- ══════════════ TAB 2 — NEXT SEASON ═════════════════ -->
      @if (selectedTab() === 'next') {
        <div class="p-4 sm:p-6">
          <div class="mb-4">
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-55"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
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
                  <li class="flex items-center gap-3 px-4 py-3" style="border-color:var(--color-border)">
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
                      <span class="font-bold text-sm" style="color:var(--color-accent)">
                        {{ p.predictedNextFantavoto | number:'1.2-2' }}
                      </span>
                    </div>
                  </li>
                }
              </ul>
            </div>
          }
        </div>
      }

      <!-- ══════════════ TAB 3 — PIPELINE INFO ═══════════════ -->
      @if (selectedTab() === 'pipeline') {
        <div class="p-4 sm:p-6 space-y-4">
          <!-- Model info -->
          @if (pipelineMeta(); as m) {
            <div class="rounded-lg border px-4 py-3 text-xs"
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

          <!-- Admin config panel -->
          @if (config(); as cfg) {
            <div class="rounded-lg border p-4"
                 style="border-color:var(--color-border);background:var(--color-surface)">
              <h3 class="text-sm font-semibold mb-3" style="color:var(--color-text-primary)">Hybrid Configuration</h3>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label class="block text-xs mb-1" style="color:var(--color-text-secondary)">
                    MANTRA weight: {{ cfg.PESO_MANTRA | percent }}
                  </label>
                  <input type="range" min="0" max="100" step="1"
                         class="w-full"
                         [value]="cfg.PESO_MANTRA * 100"
                         (input)="onSliderChange($event)" />
                  <label class="block text-xs mt-1" style="color:var(--color-text-secondary)">
                    ML weight: {{ cfg.PESO_ML | percent }}
                  </label>
                </div>
                <div class="space-y-2 text-xs" style="color:var(--color-text-secondary)">
                  <label>Prediction STD weight: <strong>{{ cfg.W_PREDICTION_STD }}</strong></label>
                  <label>Minutes weight: <strong>{{ cfg.W_MINUTES }}</strong></label>
                  <label>EV Scale Factor: <strong>{{ cfg.EV_SCALE_FACTOR }}</strong></label>
                  <label>Confidence min: <strong>{{ cfg.SOGLIA_CONFIDENZA_MIN }}</strong></label>
                  <label>Gap alert: <strong>{{ cfg.SOGLIA_GAP_ALERT }}</strong></label>
                </div>
              </div>
              <div class="mt-3 flex gap-2">
                <button class="rounded-lg border px-4 py-1.5 text-xs font-medium"
                        style="background:var(--color-accent);color:#fff;border-color:transparent"
                        (click)="saveAndRegenerate(cfg)">
                  Salva e Rigenera
                </button>
                <button class="rounded-lg border px-4 py-1.5 text-xs font-medium"
                        style="background:var(--color-surface-raised);color:var(--color-text-primary);border-color:var(--color-border)"
                        (click)="previewConfig(cfg)">
                  Prova
                </button>
              </div>
              @if (previewMessage()) {
                <div class="mt-2 rounded border px-3 py-2 text-xs"
                     [style]="previewOk()
                       ? 'border-color:#22c55e44;background:#22c55e11;color:#22c55e'
                       : 'border-color:#dc262644;background:#dc262611;color:#dc2626'">
                  {{ previewMessage() }}
                </div>
              }
              @if (lastGenerated(); as ts) {
                <p class="mt-2 text-xs" style="color:var(--color-text-secondary)">
                  Ultima rigenerazione: {{ ts | date:'d MMM y, HH:mm:ss' }}
                </p>
              }
            </div>
          } @else if (configLoading()) {
            <div class="rounded-lg border p-4" style="border-color:var(--color-border);background:var(--color-surface)">
              <app-skeleton height="80px" />
            </div>
          }
        </div>
      }
    </div>
  `,
})
export class PredictionsComponent {
  private readonly predService = inject(PredictionService);

  readonly tabs = [
    { id: 'hybrid' as const, label: 'Ibrido' },
    { id: 'next' as const, label: 'Next Season' },
    { id: 'pipeline' as const, label: 'Pipeline Info' },
  ];
  readonly skeletonRows = Array.from({ length: 8 });
  readonly selectedTab = signal<'hybrid' | 'next' | 'pipeline'>('hybrid');

  // ── Hybrid tab state ──────────────────────────────────
  readonly hybridItems = signal<HybridPlayerPrediction[]>([]);
  readonly hybridStats = signal<HybridStatsResponse | null>(null);
  readonly hybridLoading = signal(true);
  readonly hybridError = signal<string | null>(null);
  readonly hybridSearch = signal('');
  readonly hybridRuolo = signal('');
  readonly hybridConfidenceMin = signal<number | null>(null);
  readonly hybridPage = signal(1);
  readonly hybridPageSize = signal(50);
  readonly sortField = signal<string | null>(null);
  readonly sortDir = signal<'asc' | 'desc'>('desc');
  readonly activeLabels = signal(new Set<string>());
  readonly selectedPlayer = signal<HybridPlayerPrediction | null>(null);
  readonly lastGenerated = signal<string | null>(null);

  readonly confidencePresets = [
    { label: 'All', value: null as number | null },
    { label: '≥70', value: 70 },
    { label: '≥50', value: 50 },
    { label: '<30', value: -1 },
  ];

  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.filteredHybrid().length / this.hybridPageSize())));

  readonly filteredHybrid = computed(() => {
    let items = this.hybridItems();
    const q = this.hybridSearch().toLowerCase();
    const ruolo = this.hybridRuolo();
    const confMin = this.hybridConfidenceMin();
    const labels = this.activeLabels();

    if (q) items = items.filter(p => p.playerName.toLowerCase().includes(q));
    if (ruolo) items = items.filter(p => p.ruoloPrimario === ruolo);
    if (confMin !== null) {
      if (confMin === -1) items = items.filter(p => (p.confidenceScore ?? 100) < 30);
      else items = items.filter(p => (p.confidenceScore ?? 0) >= confMin);
    }
    if (labels.size > 0) {
      items = items.filter(p => p.hybridLabels.some(l => labels.has(l)));
    }

    const sf = this.sortField();
    const sd = this.sortDir();
    if (sf) {
      items = [...items].sort((a, b) => {
        const av = (a as any)[sf] ?? -999999;
        const bv = (b as any)[sf] ?? -999999;
        return sd === 'desc' ? bv - av : av - bv;
      });
    }

    return items;
  });

  // ── Next season tab state ─────────────────────────────
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

  // ── Pipeline tab state ────────────────────────────────
  readonly pipelineMeta = signal<{
    runId: string;
    bestModel: string;
    rolePartitioned: boolean;
    modelComparison: ModelComparison[];
  } | null>(null);
  readonly config = signal<HybridConfig | null>(null);
  readonly configLoading = signal(true);
  readonly previewMessage = signal<string | null>(null);
  readonly previewOk = signal(false);
  private pendingOverrides: Partial<HybridConfig> = {};

  // ── Constants (for template) ──────────────────────────
  readonly MANTRA_ROLES = MANTRA_ROLES;
  readonly HYBRID_LABELS = HYBRID_LABELS;

  // ── Methods ──────────────────────────────────────────

  scorePct(v: number): number {
    return Math.max(0, Math.min(100, ((v - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100));
  }

  labelColor(label: string): string | null {
    const found = HYBRID_LABELS.find(l => l.id === label);
    return found ? found.color : null;
  }

  setSort(field: string) {
    if (this.sortField() === field) {
      this.sortDir.set(this.sortDir() === 'desc' ? 'asc' : 'desc');
    } else {
      this.sortField.set(field);
      this.sortDir.set('desc');
    }
  }

  toggleLabel(id: string) {
    this.activeLabels.update(s => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  onSliderChange(event: Event) {
    const v = parseInt((event.target as HTMLInputElement).value, 10);
    const current = this.config();
    if (current) {
      const updated = { ...current, PESO_MANTRA: v / 100, PESO_ML: (100 - v) / 100 };
      this.config.set(updated);
      this.pendingOverrides = { PESO_MANTRA: updated.PESO_MANTRA, PESO_ML: updated.PESO_ML };
    }
  }

  saveAndRegenerate(cfg: HybridConfig) {
    const overrides = Object.keys(this.pendingOverrides).length > 0 ? this.pendingOverrides : undefined;
    this.predService.runHybrid(2025, overrides as any, true).subscribe({
      next: res => {
        this.lastGenerated.set(res.generatedAt);
        this.previewMessage.set('Config saved and results regenerated.');
        this.previewOk.set(true);
        this.pendingOverrides = {};
        this.loadHybridData();
        this.loadConfig();
      },
      error: err => {
        this.previewMessage.set('Error: ' + (err.error?.detail || err.message));
        this.previewOk.set(false);
      },
    });
  }

  previewConfig(cfg: HybridConfig) {
    const overrides = Object.keys(this.pendingOverrides).length > 0 ? this.pendingOverrides : undefined;
    this.predService.runHybrid(2025, overrides as any, false).subscribe({
      next: res => {
        this.lastGenerated.set(res.generatedAt);
        this.previewMessage.set(`Preview regenerated (${res.nPlayers} players). Not published.`);
        this.previewOk.set(true);
      },
      error: err => {
        this.previewMessage.set('Error: ' + (err.error?.detail || err.message));
        this.previewOk.set(false);
      },
    });
  }

  private loadHybridData() {
    this.hybridLoading.set(true);
    this.hybridError.set(null);
    this.predService.getHybridPredictions({
      page: this.hybridPage(),
      size: this.hybridPageSize(),
      sortBy: this.sortField() || undefined,
      sortDir: this.sortDir() === 'desc' ? 'desc' : 'asc',
    }).subscribe({
      next: res => {
        this.hybridItems.set(res.items);
        this.hybridLoading.set(false);
        const ts = res.meta?.generatedAt;
        if (ts) this.lastGenerated.set(ts);
      },
      error: () => {
        this.hybridError.set('Could not load hybrid predictions.');
        this.hybridLoading.set(false);
      },
    });

    this.predService.getHybridStats().subscribe({
      next: s => this.hybridStats.set(s),
    });
  }

  private loadConfig() {
    this.configLoading.set(true);
    this.predService.getHybridConfig().subscribe({
      next: c => {
        this.config.set(c);
        this.configLoading.set(false);
      },
    });
  }

  constructor() {
    // Load hybrid data on init
    this.loadHybridData();
    this.loadConfig();

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

    // Load pipeline meta
    this.predService.getPredictions().subscribe({
      next: res => {
        this.pipelineMeta.set({
          runId: res.runId,
          bestModel: res.bestModel,
          rolePartitioned: res.rolePartitioned,
          modelComparison: res.modelComparison,
        });
      },
    });
  }
}
