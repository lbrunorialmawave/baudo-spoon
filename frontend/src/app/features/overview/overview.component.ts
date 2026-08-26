import { Component, DestroyRef, computed, effect, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import { OverviewService } from '../../core/services/overview.service';
import { MantraService } from '../../core/services/mantra.service';
import { OverviewPlayer, SortKey } from '../../core/models/overview.models';
import { FASE7_LABELS, FASE7_AXIS, FASE7_TOOLTIPS, HYBRID_LABELS, MANTRA_ROLES } from '../../core/models/mantra.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { OverviewTableComponent } from './components/overview-table/overview-table.component';
import { OverviewDrawerComponent } from './components/overview-drawer/overview-drawer.component';

/** Unified "Players + Predictions" view: MANTRA pillars, Hybrid ML, Gruppo
 *  Esperti, and the 3 titolarità signals, one row per player. Server-side
 *  filter/sort/pagination via GET /overview/players — no whole-artifact
 *  client-side loading (unlike the Predictions "Ibrido" tab today). */
@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [FormsModule, ErrorBoundaryComponent, OverviewTableComponent, OverviewDrawerComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <div class="flex flex-col gap-1 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <div class="flex items-center gap-3">
          <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Overview Giocatori</h1>
          <span class="rounded-full px-2 py-0.5 text-xs font-medium"
                style="background:var(--color-surface-raised);color:var(--color-text-secondary)">MANTRA + ML + Esperti</span>
        </div>
        @if (total()) {
          <span class="text-xs" style="color:var(--color-text-secondary)">{{ total() }} players</span>
        }
      </div>

      <!-- Primary filters — the ones used on every visit -->
      <div class="flex flex-wrap items-center gap-2 border-b px-4 pt-3 pb-2.5 sm:px-6"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-44 md:w-52"
               style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
               placeholder="Search player" [ngModel]="searchDraft()" (ngModelChange)="onSearchChange($event)" />

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-auto"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedRuolo()" (ngModelChange)="selectedRuolo.set($event)">
          <option value="">All roles</option>
          @for (r of MANTRA_ROLES; track r) { <option [value]="r">{{ r }}</option> }
        </select>

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-auto"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedTeam()" (ngModelChange)="selectedTeam.set($event)">
          <option value="">All teams</option>
          @for (t of teamsList(); track t) { <option [value]="t">{{ t }}</option> }
        </select>

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-auto"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedFase7()" (ngModelChange)="selectedFase7.set($event)">
          <option value="">All classifications</option>
          @for (key of FASE7_KEYS; track key) {
            <option [value]="key">{{ FASE7_LABELS[key].icon }} {{ FASE7_LABELS[key].label }}</option>
          }
        </select>

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-auto"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedStatus()" (ngModelChange)="selectedStatus.set($event)"
                title="Filtra per titolarità reale (probabili formazioni)">
          <option value="">All statuses</option>
          <option value="starter">🟢 Titolare</option>
          <option value="bench">⚪ Panchina</option>
          <option value="injured">🔴 Infortunato</option>
          <option value="suspended">🔴 Squalificato</option>
          <option value="doubtful">🟡 In dubbio</option>
        </select>

        <div class="flex items-center gap-1 rounded-lg border px-2 py-1"
             style="background:var(--color-surface-raised);border-color:var(--color-border)">
          <span class="pl-0.5 text-xs" style="color:var(--color-text-secondary)">Prezzo</span>
          <input class="w-14 bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)"
                 type="number" min="0" placeholder="min"
                 [ngModel]="priceMin()" (ngModelChange)="priceMin.set($event)" />
          <span style="color:var(--color-text-secondary)">–</span>
          <input class="w-14 bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)"
                 type="number" min="0" placeholder="max"
                 [ngModel]="priceMax()" (ngModelChange)="priceMax.set($event)" />
        </div>

        <div class="flex items-center gap-1 rounded-lg border px-2 py-1"
             style="background:var(--color-surface-raised);border-color:var(--color-border)"
             title="Filtra per punteggio Gruppo Esperti (TOTALE su 50)">
          <span class="pl-0.5 text-xs" style="color:var(--color-text-secondary)">Esperti</span>
          <input class="w-12 bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)"
                 type="number" min="0" max="50" placeholder="min"
                 [ngModel]="expertMin()" (ngModelChange)="expertMin.set($event)" />
          <span style="color:var(--color-text-secondary)">–</span>
          <input class="w-12 bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)"
                 type="number" min="0" max="50" placeholder="max"
                 [ngModel]="expertMax()" (ngModelChange)="expertMax.set($event)" />
        </div>

        <button class="rounded-lg border px-2.5 py-1.5 text-xs font-medium"
                style="border-color:var(--color-border);color:var(--color-text-secondary)"
                (click)="clearFilters()">Clear</button>
      </div>

      <!-- Advanced filters — collapsed by default, every secondary dimension -->
      <div class="border-b sm:px-6" style="border-color:var(--color-border);background:var(--color-surface)">
        <button class="flex w-full items-center gap-2 px-4 py-2 text-xs font-medium sm:px-0"
                style="color:var(--color-text-secondary)"
                [attr.aria-expanded]="advancedOpen()"
                (click)="advancedOpen.set(!advancedOpen())">
          <span class="transition-transform" [style.transform]="advancedOpen() ? 'rotate(90deg)' : 'none'">▸</span>
          Filtri avanzati
          @if (advancedActiveCount() > 0) {
            <span class="rounded-full px-1.5 py-0.5 text-[10px] font-semibold text-white" style="background:var(--color-accent)">
              {{ advancedActiveCount() }}
            </span>
          }
        </button>

        @if (advancedOpen()) {
          <div class="px-4 pb-4 sm:px-0">
            <div class="grid gap-3" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">FP Ibrido</span>
                <div class="flex items-center gap-1 rounded-lg border px-2 py-1" style="background:var(--color-surface-raised);border-color:var(--color-border)">
                  <input class="w-full bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)" type="number" placeholder="min"
                         [ngModel]="fpIbridoMin()" (ngModelChange)="fpIbridoMin.set($event)" />
                  <span style="color:var(--color-text-secondary)">–</span>
                  <input class="w-full bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)" type="number" placeholder="max"
                         [ngModel]="fpIbridoMax()" (ngModelChange)="fpIbridoMax.set($event)" />
                </div>
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">FP Mantra</span>
                <div class="flex items-center gap-1 rounded-lg border px-2 py-1" style="background:var(--color-surface-raised);border-color:var(--color-border)">
                  <input class="w-full bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)" type="number" min="0" max="100" placeholder="min"
                         [ngModel]="fpMantraMin()" (ngModelChange)="fpMantraMin.set($event)" />
                  <span style="color:var(--color-text-secondary)">–</span>
                  <input class="w-full bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)" type="number" min="0" max="100" placeholder="max"
                         [ngModel]="fpMantraMax()" (ngModelChange)="fpMantraMax.set($event)" />
                </div>
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">VR</span>
                <div class="flex items-center gap-1 rounded-lg border px-2 py-1" style="background:var(--color-surface-raised);border-color:var(--color-border)">
                  <input class="w-full bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)" type="number" min="0" placeholder="min"
                         [ngModel]="vrMin()" (ngModelChange)="vrMin.set($event)" />
                  <span style="color:var(--color-text-secondary)">–</span>
                  <input class="w-full bg-transparent px-1 py-0.5 text-sm outline-none" style="color:var(--color-text-primary)" type="number" min="0" placeholder="max"
                         [ngModel]="vrMax()" (ngModelChange)="vrMax.set($event)" />
                </div>
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Confidence ML minima</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="100" placeholder="es. 60"
                       [ngModel]="confidenceMin()" (ngModelChange)="confidenceMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Titolarità ML minima (%)</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="100" placeholder="es. 70"
                       [ngModel]="startProbMin()" (ngModelChange)="startProbMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Titolarità reale minima (%)</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="100" placeholder="es. 70"
                       [ngModel]="probScrapedMin()" (ngModelChange)="probScrapedMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Esperti — rating minimo (★)</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="5" placeholder="1-5"
                       [ngModel]="expertRatingMin()" (ngModelChange)="expertRatingMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Esperti — titolarità minima</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="10" placeholder="1-10"
                       [ngModel]="expertTitolaritaMin()" (ngModelChange)="expertTitolaritaMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Esperti — media voto minima</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="10" placeholder="1-10"
                       [ngModel]="expertMediaVotoMin()" (ngModelChange)="expertMediaVotoMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Esperti — salute minima</span>
                <input class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                       type="number" min="0" max="10" placeholder="1-10"
                       [ngModel]="expertSaluteMin()" (ngModelChange)="expertSaluteMin.set($event)" />
              </label>

              <label class="flex flex-col gap-1">
                <span class="text-xs" style="color:var(--color-text-secondary)">Dati ML</span>
                <select class="rounded-lg border px-2 py-1.5 text-sm outline-none w-full" style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                        [ngModel]="hasMlDataFilter()" (ngModelChange)="hasMlDataFilter.set($event)">
                  <option value="">Tutti</option>
                  <option value="yes">Solo con dati ML</option>
                  <option value="no">Solo senza dati ML</option>
                </select>
              </label>

              <label class="flex items-center gap-2 self-end pb-1.5" title="Oggi il flag copre solo 'Cambio Squadra'">
                <input type="checkbox" [ngModel]="hasRiskFlag()" (ngModelChange)="hasRiskFlag.set($event)" />
                <span class="text-xs" style="color:var(--color-text-secondary)">Solo con flag di rischio contestuale</span>
              </label>
            </div>

            <div class="mt-3 flex flex-wrap items-center gap-1.5">
              <span class="text-xs mr-1" style="color:var(--color-text-secondary)">Segnali ML:</span>
              @for (l of HYBRID_LABELS; track l.id) {
                <button class="rounded-full border px-2.5 py-0.5 text-xs font-medium transition-all"
                        [style]="activeLabels().has(l.id)
                          ? 'background:' + l.color + ';color:#fff;border-color:transparent'
                          : 'background:var(--color-surface-raised);color:var(--color-text-secondary);border-color:var(--color-border);opacity:0.7'"
                        [title]="l.desc"
                        (click)="toggleLabel(l.id)">
                  {{ l.label }}
                </button>
              }
            </div>
          </div>
        }
      </div>

      <!-- Sort summary — independent from filters: "Clear" above never touches
           this, "Cancella ordinamento" here never touches filters (see
           clearFilters()/clearSort()). -->
      @if (sortKeys().length > 0) {
        <div class="flex flex-wrap items-center gap-2 border-b px-4 py-2 sm:px-6"
             style="border-color:var(--color-border);background:var(--color-surface)">
          <span class="text-xs" style="color:var(--color-text-secondary)">Ordinamento:</span>
          @for (k of sortKeys(); track k.column; let i = $index) {
            <span class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs"
                  style="border-color:var(--color-border);background:var(--color-surface-raised);color:var(--color-text-primary)">
              {{ i + 1 }}. {{ sortColumnLabel(k.column) }} {{ k.direction === 'asc' ? '↑' : '↓' }}
              <button class="ml-0.5 opacity-60 hover:opacity-100" (click)="removeSortKey(k.column)" aria-label="Rimuovi criterio">✕</button>
            </span>
          }
          @if (sortKeys().length > 1) {
            <button class="text-xs underline" style="color:var(--color-text-secondary)" (click)="clearSort()">
              Cancella ordinamento
            </button>
          }
        </div>
      }

      <div class="p-4 sm:p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else {
          <div class="card p-0 overflow-hidden">
            <app-overview-table
              [items]="players()"
              [loading]="loading()"
              [page]="currentPage()"
              [pageSize]="pageSize"
              [sortKeys]="sortKeys()"
              (sortChanged)="onSort($event)"
              (playerSelected)="selectedPlayer.set($event)" />
          </div>
          @let displayPages = totalPages();
          @if (displayPages > 1) {
            <div class="mt-4 flex items-center justify-between text-sm" style="color:var(--color-text-secondary)">
              <span>Page {{ currentPage() }} of {{ displayPages }}</span>
              <div class="flex gap-2">
                <button class="rounded-lg border px-3 py-1.5 text-xs" style="border-color:var(--color-border)"
                        [disabled]="currentPage() <= 1" (click)="currentPage.update(p => p - 1)">Prev</button>
                <button class="rounded-lg border px-3 py-1.5 text-xs" style="border-color:var(--color-border)"
                        [disabled]="currentPage() >= displayPages" (click)="currentPage.update(p => p + 1)">Next</button>
              </div>
            </div>
          }
        }
      </div>
    </div>
    @if (selectedPlayer(); as p) {
      <app-overview-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
  `,
})
export class OverviewComponent {
  private readonly overviewService = inject(OverviewService);
  private readonly mantraService = inject(MantraService);
  private readonly destroyRef = inject(DestroyRef);

  readonly teamsList = signal<string[]>([]);

  readonly MANTRA_ROLES = MANTRA_ROLES;
  readonly FASE7_LABELS = FASE7_LABELS;
  readonly FASE7_KEYS = Object.keys(FASE7_LABELS);
  readonly FASE7_TOOLTIPS = FASE7_TOOLTIPS;
  readonly HYBRID_LABELS = HYBRID_LABELS;

  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly players = signal<OverviewPlayer[]>([]);
  readonly total = signal(0);
  readonly currentPage = signal(1);
  readonly pageSize = 50;

  // ── Primary filters ──────────────────────────────────────────────
  readonly selectedRuolo = signal('');
  readonly selectedTeam = signal('');
  readonly selectedFase7 = signal('');
  readonly selectedStatus = signal('');
  readonly searchDraft = signal('');
  readonly searchInput = signal('');
  readonly priceMin = signal<number | null>(null);
  readonly priceMax = signal<number | null>(null);
  readonly expertMin = signal<number | null>(null);
  readonly expertMax = signal<number | null>(null);

  // ── Advanced filters ─────────────────────────────────────────────
  readonly advancedOpen = signal(false);
  readonly activeLabels = signal<Set<string>>(new Set());
  readonly fpIbridoMin = signal<number | null>(null);
  readonly fpIbridoMax = signal<number | null>(null);
  readonly fpMantraMin = signal<number | null>(null);
  readonly fpMantraMax = signal<number | null>(null);
  readonly vrMin = signal<number | null>(null);
  readonly vrMax = signal<number | null>(null);
  readonly confidenceMin = signal<number | null>(null);
  readonly startProbMin = signal<number | null>(null);
  readonly probScrapedMin = signal<number | null>(null);
  readonly expertRatingMin = signal<number | null>(null);
  readonly expertTitolaritaMin = signal<number | null>(null);
  readonly expertMediaVotoMin = signal<number | null>(null);
  readonly expertSaluteMin = signal<number | null>(null);
  readonly hasMlDataFilter = signal<'' | 'yes' | 'no'>('');
  readonly hasRiskFlag = signal(false);

  readonly advancedActiveCount = computed(() => {
    let n = 0;
    if (this.activeLabels().size) n += this.activeLabels().size;
    if (this.fpIbridoMin() != null) n++;
    if (this.fpIbridoMax() != null) n++;
    if (this.fpMantraMin() != null) n++;
    if (this.fpMantraMax() != null) n++;
    if (this.vrMin() != null) n++;
    if (this.vrMax() != null) n++;
    if (this.confidenceMin() != null) n++;
    if (this.startProbMin() != null) n++;
    if (this.probScrapedMin() != null) n++;
    if (this.expertRatingMin() != null) n++;
    if (this.expertTitolaritaMin() != null) n++;
    if (this.expertMediaVotoMin() != null) n++;
    if (this.expertSaluteMin() != null) n++;
    if (this.hasMlDataFilter()) n++;
    if (this.hasRiskFlag()) n++;
    return n;
  });

  readonly selectedPlayer = signal<OverviewPlayer | null>(null);

  // ── Multi-column sort — independent of the filter signals above ────
  readonly MAX_SORT_KEYS = 3; // must mirror api/src/routers/overview.py
  readonly sortKeys = signal<SortKey[]>([]);
  readonly sortByParam = computed(() =>
    this.sortKeys().map(k => (k.direction === 'desc' ? '-' : '') + k.column).join(',') || undefined
  );

  private static readonly SORT_COLUMN_LABELS: Record<string, string> = {
    player_name: 'Player',
    team: 'Team',
    ruolo_primario: 'Ruolo',
    fpIbrido: 'FP Ibrido',
    FP_Mantra: 'FP Mantra',
    predicted_fantavoto: 'Voto ML',
    confidenceScore: 'Conf.',
    VR: 'VR',
    Pz1: 'Prezzo',
    expert_totale: 'Esperti',
  };
  readonly sortColumnLabel = (column: string): string =>
    OverviewComponent.SORT_COLUMN_LABELS[column] ?? column;

  private readonly searchQuery$ = new Subject<string>();
  private lastFilterSignature = '';

  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize)));

  constructor() {
    this.searchQuery$
      .pipe(debounceTime(300), distinctUntilChanged(), takeUntilDestroyed(this.destroyRef))
      .subscribe(v => this.searchInput.set(v));

    effect(() => {
      const signature = JSON.stringify([
        this.selectedRuolo(),
        this.selectedTeam(),
        this.selectedFase7(),
        this.selectedStatus(),
        Array.from(this.activeLabels()).sort(),
        this.searchInput(),
        this.priceMin(),
        this.priceMax(),
        this.expertMin(),
        this.expertMax(),
        this.fpIbridoMin(),
        this.fpIbridoMax(),
        this.fpMantraMin(),
        this.fpMantraMax(),
        this.vrMin(),
        this.vrMax(),
        this.confidenceMin(),
        this.startProbMin(),
        this.probScrapedMin(),
        this.expertRatingMin(),
        this.expertTitolaritaMin(),
        this.expertMediaVotoMin(),
        this.expertSaluteMin(),
        this.hasMlDataFilter(),
        this.hasRiskFlag(),
      ]);
      const filtersChanged = signature !== this.lastFilterSignature;
      this.lastFilterSignature = signature;
      if (filtersChanged && this.currentPage() !== 1) {
        this.currentPage.set(1);
        return;
      }

      this.currentPage();
      this.sortKeys();
      this.loadData();
    });

    this.mantraService.getTeams().subscribe({
      next: (res) => this.teamsList.set(res.teams),
      error: () => {},
    });
  }

  readonly onSearchChange = (value: string) => {
    this.searchDraft.set(value);
    this.searchQuery$.next(value);
  };

  private loadData(): void {
    this.loading.set(true);
    this.error.set(null);

    const mlFilter = this.hasMlDataFilter();
    const fase7Selected = this.selectedFase7() || undefined;
    const fase7Axis = fase7Selected ? FASE7_AXIS[fase7Selected] : undefined;

    this.overviewService.listPlayers({
      ruolo: this.selectedRuolo() || undefined,
      team: this.selectedTeam() || undefined,
      fase7Rendimento: fase7Axis === 'rendimento' ? fase7Selected : undefined,
      fase7Prezzo: fase7Axis === 'prezzo' ? fase7Selected : undefined,
      labels: this.activeLabels().size ? Array.from(this.activeLabels()) : undefined,
      search: this.searchInput() || undefined,
      minPrice: this.priceMin() ?? undefined,
      maxPrice: this.priceMax() ?? undefined,
      statusScraped: this.selectedStatus() || undefined,
      expertTotaleMin: this.expertMin() ?? undefined,
      expertTotaleMax: this.expertMax() ?? undefined,
      minFpIbrido: this.fpIbridoMin() ?? undefined,
      maxFpIbrido: this.fpIbridoMax() ?? undefined,
      minFp: this.fpMantraMin() ?? undefined,
      maxFp: this.fpMantraMax() ?? undefined,
      minVr: this.vrMin() ?? undefined,
      maxVr: this.vrMax() ?? undefined,
      confidenceMin: this.confidenceMin() ?? undefined,
      startProbabilityMin: this.startProbMin() ?? undefined,
      probabilityScrapedMin: this.probScrapedMin() ?? undefined,
      expertRatingMin: this.expertRatingMin() ?? undefined,
      expertTitolaritaMin: this.expertTitolaritaMin() ?? undefined,
      expertMediaVotoMin: this.expertMediaVotoMin() ?? undefined,
      expertSaluteMin: this.expertSaluteMin() ?? undefined,
      hasMlData: mlFilter === 'yes' ? true : mlFilter === 'no' ? false : undefined,
      hasRiskFlag: this.hasRiskFlag() ? true : undefined,
      sortBy: this.sortByParam(),
      page: this.currentPage(),
      size: this.pageSize,
    }).subscribe({
      next: (res) => {
        this.players.set(res.items);
        this.total.set(res.total);
        this.loading.set(false);
      },
      error: (e) => {
        this.error.set(this.extractErrorMessage(e));
        this.loading.set(false);
      },
    });
  }

  private extractErrorMessage(e: unknown): string {
    const detail = (e as { error?: { detail?: unknown } } | null)?.error?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: any) => d?.msg ? `${(d.loc ?? []).join('.')}: ${d.msg}` : JSON.stringify(d))
        .join('; ');
    }
    return (e as { message?: string } | null)?.message ?? 'Failed to load';
  }

  readonly toggleLabel = (id: string) => {
    this.activeLabels.update(s => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  /** Plain click = new sole sort key (toggles direction if it's already
   *  the only one). Shift+click = add/toggle a secondary/tertiary key
   *  without losing the others (Excel/Airtable convention), replacing the
   *  lowest-priority key once MAX_SORT_KEYS is reached — always visible
   *  feedback, never a silently-ignored click. */
  readonly onSort = ({ column, additive }: { column: string; additive: boolean }) => {
    this.sortKeys.update(keys => {
      const idx = keys.findIndex(k => k.column === column);

      if (!additive) {
        if (keys.length === 1 && idx === 0) {
          return [{ column, direction: keys[0].direction === 'asc' ? 'desc' : 'asc' }];
        }
        return [{ column, direction: 'asc' }];
      }

      if (idx !== -1) {
        const next = [...keys];
        next[idx] = { ...next[idx], direction: next[idx].direction === 'asc' ? 'desc' : 'asc' };
        return next;
      }

      if (keys.length >= this.MAX_SORT_KEYS) {
        return [...keys.slice(0, -1), { column, direction: 'asc' as const }];
      }
      return [...keys, { column, direction: 'asc' as const }];
    });
    this.currentPage.set(1);
  };

  readonly removeSortKey = (column: string) => {
    this.sortKeys.update(keys => keys.filter(k => k.column !== column));
    this.currentPage.set(1);
  };

  readonly clearSort = () => {
    this.sortKeys.set([]);
    this.currentPage.set(1);
  };

  readonly clearFilters = () => {
    this.selectedRuolo.set('');
    this.selectedTeam.set('');
    this.selectedFase7.set('');
    this.selectedStatus.set('');
    this.searchDraft.set('');
    this.searchInput.set('');
    this.priceMin.set(null);
    this.priceMax.set(null);
    this.expertMin.set(null);
    this.expertMax.set(null);
    this.activeLabels.set(new Set());
    this.fpIbridoMin.set(null);
    this.fpIbridoMax.set(null);
    this.fpMantraMin.set(null);
    this.fpMantraMax.set(null);
    this.vrMin.set(null);
    this.vrMax.set(null);
    this.confidenceMin.set(null);
    this.startProbMin.set(null);
    this.probScrapedMin.set(null);
    this.expertRatingMin.set(null);
    this.expertTitolaritaMin.set(null);
    this.expertMediaVotoMin.set(null);
    this.expertSaluteMin.set(null);
    this.hasMlDataFilter.set('');
    this.hasRiskFlag.set(false);
  };
}
