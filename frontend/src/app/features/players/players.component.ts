import { Component, DestroyRef, computed, effect, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import { StatsService } from '../../core/services/stats.service';
import { MantraService } from '../../core/services/mantra.service';
import { TeamStrengthService } from '../../core/services/team-strength.service';
import { QuotationService } from '../../core/services/quotation.service';
import { ExpertRatingsService } from '../../core/services/expert-ratings.service';
import { ExpertRatingWithFantacalcioId } from '../../core/models/expert-ratings.models';
import { FASE7_LABELS, FASE7_TOOLTIPS, MANTRA_ROLES, MantraPlayer } from '../../core/models/mantra.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { PlayerTableComponent } from './components/player-table/player-table.component';
import { PlayerDrawerComponent } from './components/player-drawer/player-drawer.component';

@Component({
  selector: 'app-players',
  standalone: true,
  imports: [FormsModule, DecimalPipe, ErrorBoundaryComponent, PlayerTableComponent, PlayerDrawerComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <div class="flex flex-col gap-1 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <div class="flex items-center gap-3">
          <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Players</h1>
          <span class="rounded-full px-2 py-0.5 text-xs font-medium"
                style="background:var(--color-surface-raised);color:var(--color-text-secondary)">MANTRA</span>
        </div>
        @if (total()) {
          <span class="text-xs" style="color:var(--color-text-secondary)">{{ total() }} players</span>
        }
      </div>

      <!-- Primary filters — always visible, tight grouping (the controls used on every visit). -->
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
                [ngModel]="selectedFase7()" (ngModelChange)="onFase7Change($event)">
          <option value="">All classifications</option>
          @for (key of FASE7_KEYS; track key) {
            <option [value]="key">{{ FASE7_LABELS[key].icon }} {{ FASE7_LABELS[key].label }}</option>
          }
        </select>

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-auto"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedStatus()" (ngModelChange)="selectedStatus.set($event)">
          <option value="">All statuses</option>
          <option value="starter">🟢 Titolare</option>
          <option value="bench">⚪ Panchina</option>
          <option value="injured">🔴 Infortunato</option>
          <option value="suspended">🔴 Squalificato</option>
          <option value="doubtful">🟡 In dubbio</option>
        </select>

        <!-- Price range grouped as one compound control, not two loose inputs. -->
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
      </div>

      <!-- Quick filters + reset — secondary row, visually lighter than the primary filters above. -->
      <div class="flex flex-wrap items-center gap-1.5 border-b px-4 py-2 sm:px-6"
           style="border-color:var(--color-border);background:var(--color-surface)">
        @for (qf of quickFilters; track qf.key) {
          <button class="rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors"
                  [style.border-color]="activeQuickFilter() === qf.key ? (FASE7_LABELS[qf.key]?.color ?? 'var(--color-accent)') : 'var(--color-border)'"
                  [style.color]="activeQuickFilter() === qf.key ? (FASE7_LABELS[qf.key]?.color ?? 'var(--color-accent)') : 'var(--color-text-secondary)'"
                  [title]="FASE7_TOOLTIPS[qf.key]"
                  (click)="toggleQuickFilter(qf.key)">
            {{ qf.icon }} {{ qf.label }}
          </button>
        }
        <button class="rounded-lg border px-2.5 py-1 text-xs font-medium"
                style="border-color:var(--color-border);color:var(--color-text-secondary)"
                (click)="clearFilters()">Clear</button>
      </div>

      <!-- Stima d'asta — optional/advanced tool, demoted to its own panel so it doesn't
           compete visually with the always-used filters above. -->
      <div class="border-b px-4 py-2.5 sm:px-6" style="border-color:var(--color-border);background:var(--color-bg)">
        <div class="flex flex-wrap items-center gap-3">
          <label class="inline-flex cursor-pointer items-center gap-2 text-xs font-medium select-none"
                 style="color:var(--color-text-primary)"
                 title="Stima il prezzo massimo d'asta in base al percentile del giocatore nel ruolo e al numero di partecipanti alla lega, invece di mostrare solo la quotazione ufficiale">
            <input type="checkbox" class="sr-only" [ngModel]="stimaAstaEnabled()" (ngModelChange)="stimaAstaEnabled.set($event)" />
            <span class="inline-block h-5 w-9 shrink-0 rounded-full transition-colors"
                  [style.background]="stimaAstaEnabled() ? 'var(--color-accent)' : 'var(--color-border)'">
              <span class="block h-4 w-4 rounded-full transition-transform" style="background:#fff;margin-top:2px"
                    [style.transform]="stimaAstaEnabled() ? 'translateX(18px)' : 'translateX(2px)'"></span>
            </span>
            Stima prezzo d'asta
          </label>

          @if (stimaAstaEnabled()) {
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-36"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   type="number" min="1" placeholder="N. partecipanti"
                   [ngModel]="numPartecipanti()" (ngModelChange)="numPartecipanti.set($event)" />
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-36"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   type="number" min="1" placeholder="Budget a manager"
                   [ngModel]="budget()" (ngModelChange)="budget.set($event)" />
            @if (numPartecipanti() != null && budget() != null) {
              <span class="text-xs" style="color:var(--color-text-secondary)"
                    title="Crediti totali di lega = budget a manager × numero partecipanti">
                Crediti totali lega: <strong style="color:var(--color-text-primary)">{{ numPartecipanti()! * budget()! }}</strong>
              </span>
            }
            <button class="rounded-lg border px-2.5 py-1 text-xs font-medium"
                    style="border-color:var(--color-border);color:var(--color-text-secondary)"
                    (click)="showRoleOverrides.set(!showRoleOverrides())">
              {{ showRoleOverrides() ? 'Nascondi impostazioni' : 'Impostazioni per ruolo' }} ⚙
            </button>
          } @else {
            <span class="text-xs" style="color:var(--color-text-secondary)">Mostra solo la quotazione ufficiale di listone</span>
          }
        </div>

        @if (stimaAstaEnabled() && showRoleOverrides()) {
          <div class="mt-3 overflow-x-auto rounded-lg border" style="border-color:var(--color-border);background:var(--color-surface)">
            <div class="flex items-center justify-between border-b px-3 py-2" style="border-color:var(--color-border)">
              <span class="text-xs font-medium" style="color:var(--color-text-primary)">Parametri di inflazione per gruppo di ruolo</span>
              <button class="text-xs underline-offset-2 hover:underline" style="color:var(--color-text-secondary)"
                      (click)="resetRoleOverrides()">Reimposta predefiniti</button>
            </div>
            <table class="w-full text-xs" style="border-collapse:collapse;min-width:480px">
              <thead>
                <tr style="color:var(--color-text-secondary);background:var(--color-surface-raised)">
                  <th class="px-3 py-2 text-left font-medium">Gruppo di ruolo</th>
                  <th class="px-3 py-2 text-left font-medium" title="Percentile minimo nel ruolo da cui parte l'inflazione">
                    Soglia percentile <span class="opacity-60">(0–1)</span>
                  </th>
                  <th class="px-3 py-2 text-left font-medium" title="Velocità di crescita del prezzo per partecipante oltre la baseline">
                    Tasso base
                  </th>
                  <th class="px-3 py-2 text-left font-medium" title="Moltiplicatore massimo applicabile al prezzo di listino">
                    Moltiplicatore max <span class="opacity-60">(×)</span>
                  </th>
                  <th class="px-3 py-2 text-left font-medium" title="Quota dei crediti totali di lega assegnata a questo gruppo, in base ai coefficienti di ruolo MANTRA">
                    Budget gruppo
                  </th>
                </tr>
              </thead>
              <tbody>
                @for (g of ROLE_GROUPS; track g.slug; let i = $index) {
                  <tr [style.background]="i % 2 === 1 ? 'var(--color-surface-raised)' : 'transparent'">
                    <td class="px-3 py-1.5 font-medium" style="color:var(--color-text-primary)">{{ g.label }}</td>
                    <td class="px-3 py-1.5">
                      <input class="w-20 rounded border px-2 py-1 outline-none" style="background:var(--color-bg);border-color:var(--color-border);color:var(--color-text-primary)"
                             type="number" min="0" max="1" step="0.05"
                             [ngModel]="roleOverrides()[g.slug].percentileSoglia"
                             (ngModelChange)="updateRoleOverride(g.slug, 'percentileSoglia', $event)" />
                    </td>
                    <td class="px-3 py-1.5">
                      <input class="w-20 rounded border px-2 py-1 outline-none" style="background:var(--color-bg);border-color:var(--color-border);color:var(--color-text-primary)"
                             type="number" min="0" step="0.01"
                             [ngModel]="roleOverrides()[g.slug].tassoBase"
                             (ngModelChange)="updateRoleOverride(g.slug, 'tassoBase', $event)" />
                    </td>
                    <td class="px-3 py-1.5">
                      <input class="w-20 rounded border px-2 py-1 outline-none" style="background:var(--color-bg);border-color:var(--color-border);color:var(--color-text-primary)"
                             type="number" min="1" step="0.1"
                             [ngModel]="roleOverrides()[g.slug].moltiplicatoreMax"
                             (ngModelChange)="updateRoleOverride(g.slug, 'moltiplicatoreMax', $event)" />
                    </td>
                    <td class="px-3 py-1.5" style="color:var(--color-text-secondary)">
                      {{ budgetPerGruppo()?.[g.slug] != null ? (budgetPerGruppo()![g.slug] | number:'1.0-0') + ' cr' : '—' }}
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </div>

      @if (stats(); as s) {
        <div class="flex flex-wrap gap-3 border-b px-4 py-2 text-xs sm:px-6"
             style="border-color:var(--color-border);color:var(--color-text-secondary);background:var(--color-surface)">
          <span title="Media Fantacalcio Punti di tutti i giocatori">Avg FP: <strong style="color:var(--color-accent)">{{ s.avg_fp_mantra | number:'1.1-1' }}</strong></span>
          <span title="Media Valore Reale di tutti i giocatori — indice di convenienza prezzo/valore (0-300, ~100 = valore equo)">Avg VR: <strong>{{ s.avg_vr | number:'1.0-0' }}</strong></span>
          <span class="md:ml-auto">{{ s.total_players }} scored</span>
        </div>
      }

      <div class="p-4 sm:p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else {
          <div class="card p-0 overflow-hidden">
            <app-player-table
              [items]="mantraPlayers()"
              [loading]="loading()"
              [page]="currentPage()"
              [pageSize]="pageSize"
              [stimaAstaActive]="stimaAstaActive()"
              [mantraMap]="mantraMap()"
              [matchdayStatus]="matchdayStatusMap()"
              [teamStrength]="teamStrengthScores()"
              [expertRatings]="expertRatingsMap()"
              [sortColumn]="sortColumn()"
              [sortDirection]="sortDirection()"
              (sortChanged)="onSort($any($event))"
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
      <app-player-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
  `,
})
export class PlayersComponent {
  private readonly mantraService = inject(MantraService);
  private readonly statsService = inject(StatsService);
  private readonly teamStrengthService = inject(TeamStrengthService);
  private readonly quotationService = inject(QuotationService);
  private readonly expertRatingsService = inject(ExpertRatingsService);
  private readonly destroyRef = inject(DestroyRef);

  readonly teamStrengthScores = signal<Record<string, number>>({});
  readonly teamsList = computed(() => Object.keys(this.teamStrengthScores()).sort());

  readonly MANTRA_ROLES = MANTRA_ROLES;
  readonly FASE7_LABELS = FASE7_LABELS;
  readonly FASE7_KEYS = Object.keys(FASE7_LABELS);

  readonly FASE7_TOOLTIPS = FASE7_TOOLTIPS;

  readonly quickFilters = [
    { key: 'TOP', label: 'TOP', icon: '\u{1F3C6}' },
    { key: 'AFFARE', label: 'Affari', icon: '\u{1F48E}' },
    { key: 'CERTEZZA', label: 'Certezze', icon: '✅' },
    { key: 'SCOMMESSA', label: 'Scommesse', icon: '\u{1F504}' },
  ];

  /** Macro-gruppi di ruolo per la stima d'asta — stessi slug/gruppi del backend (RUOLO_MACRO_GRUPPO). */
  readonly ROLE_GROUPS: { slug: string; label: string }[] = [
    { slug: 'portieri', label: 'Portieri' },
    { slug: 'difesa', label: 'Difesa' },
    { slug: 'ibridi', label: 'Ibridi (E, M)' },
    { slug: 'centro', label: 'Centrocampo (C)' },
    { slug: 'fantasia', label: 'Fantasia (T, W)' },
    { slug: 'attacco', label: 'Attacco' },
  ];
  private static readonly DEFAULT_ROLE_OVERRIDE = { percentileSoglia: 0.7, tassoBase: 0.05, moltiplicatoreMax: 1.6 };

  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly mantraPlayers = signal<MantraPlayer[]>([]);
  readonly stats = signal<{ avg_fp_mantra: number; avg_vr: number; total_players: number } | null>(null);
  readonly total = signal(0);
  readonly currentPage = signal(1);
  readonly pageSize = 50;
  readonly selectedRuolo = signal('');
  readonly selectedTeam = signal('');
  readonly selectedFase7 = signal('');
  readonly selectedStatus = signal('');
  /** Raw value bound to the search input, updated on every keystroke (instant UI feedback). */
  readonly searchDraft = signal('');
  /** Debounced value actually sent to the API — see the search$ subscription in the constructor. */
  readonly searchInput = signal('');
  readonly priceMin = signal<number | null>(null);
  readonly priceMax = signal<number | null>(null);
  readonly stimaAstaEnabled = signal(false);
  readonly numPartecipanti = signal<number | null>(null);
  /** Crediti per manager — cr_totali di lega = budget * numPartecipanti. */
  readonly budget = signal<number | null>(500);
  /** Riflette se l'ultima risposta del backend ha applicato la stima (può
   * essere false anche con stimaAstaEnabled=true finché numPartecipanti è vuoto). */
  readonly stimaAstaActive = signal(false);
  /** Quota di cr_totali per macro-gruppo di ruolo, dall'ultima risposta del backend. */
  readonly budgetPerGruppo = signal<Record<string, number> | null>(null);
  /** Mostra il pannello con le impostazioni avanzate per gruppo di ruolo. */
  readonly showRoleOverrides = signal(false);
  readonly roleOverrides = signal<Record<string, { percentileSoglia: number; tassoBase: number; moltiplicatoreMax: number }>>(
    PlayersComponent.defaultRoleOverrides()
  );

  private static defaultRoleOverrides(): Record<string, { percentileSoglia: number; tassoBase: number; moltiplicatoreMax: number }> {
    return {
      portieri: { ...PlayersComponent.DEFAULT_ROLE_OVERRIDE },
      difesa: { ...PlayersComponent.DEFAULT_ROLE_OVERRIDE },
      ibridi: { ...PlayersComponent.DEFAULT_ROLE_OVERRIDE },
      centro: { ...PlayersComponent.DEFAULT_ROLE_OVERRIDE },
      fantasia: { ...PlayersComponent.DEFAULT_ROLE_OVERRIDE },
      attacco: { ...PlayersComponent.DEFAULT_ROLE_OVERRIDE },
    };
  }

  readonly updateRoleOverride = (slug: string, field: 'percentileSoglia' | 'tassoBase' | 'moltiplicatoreMax', value: number) => {
    this.roleOverrides.update(current => ({
      ...current,
      [slug]: { ...current[slug], [field]: value },
    }));
  };

  readonly resetRoleOverrides = () => {
    this.roleOverrides.set(PlayersComponent.defaultRoleOverrides());
  };
  readonly activeQuickFilter = signal<string | null>(null);
  readonly selectedPlayer = signal<any | null>(null);
  readonly matchdayStatusMap = signal<Record<number, any>>({});
  readonly expertRatingsMap = signal<Record<number, ExpertRatingWithFantacalcioId>>({});
  readonly sortColumn = signal<string>('');
  readonly sortDirection = signal<'asc' | 'desc'>('asc');

  private readonly searchQuery$ = new Subject<string>();
  private lastFilterSignature = '';

  readonly mantraMap = computed(() => {
    const map: Record<number, any> = {};
    for (const mp of this.mantraPlayers()) {
      map[mp.fantacalcio_id] = mp;
    }
    return map;
  });

  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize)));

  constructor() {
    this.searchQuery$
      .pipe(
        debounceTime(300),
        distinctUntilChanged(),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(v => this.searchInput.set(v));

    effect(() => {
      // Reading every filter (not sort, not page) registers them as this
      // effect's dependencies. When any of them changes, reset to page 1 —
      // `currentPage.set(1)` below is only observable downstream if the
      // value actually changes, so this is a no-op while already on page 1.
      const signature = JSON.stringify([
        this.selectedRuolo(),
        this.selectedTeam(),
        this.selectedFase7(),
        this.activeQuickFilter(),
        this.searchInput(),
        this.selectedStatus(),
        this.priceMin(),
        this.priceMax(),
        this.matchdayStatusMap(),
        this.stimaAstaEnabled(),
        this.numPartecipanti(),
        this.budget(),
        this.roleOverrides(),
      ]);
      const filtersChanged = signature !== this.lastFilterSignature;
      this.lastFilterSignature = signature;
      if (filtersChanged && this.currentPage() !== 1) {
        this.currentPage.set(1);
        return; // the currentPage change above re-triggers this effect; loadData() runs on that pass
      }

      // Also depend on page/sort so paging and sorting reload data too.
      this.currentPage();
      this.sortColumn();
      this.sortDirection();
      this.loadData();
    });
    this.loadStats();
    this.loadMatchdayStatus();
    this.teamStrengthService.getScores().subscribe(s => this.teamStrengthScores.set(s));

    this.quotationService.getSeasons().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (seasons) => {
        const latest = seasons.length ? Math.max(...seasons) : null;
        if (latest) this.loadExpertRatings(latest);
      },
    });
  }

  readonly onSearchChange = (value: string) => {
    this.searchDraft.set(value);
    this.searchQuery$.next(value);
  };

  readonly onFase7Change = (value: string) => {
    this.selectedFase7.set(value);
    this.activeQuickFilter.set(null);
  };

  private loadData(): void {
    this.loading.set(true);
    this.error.set(null);

    const fantacalcioIds = this.selectedStatus()
      ? Object.values(this.matchdayStatusMap())
          .filter((mds: any) => mds?.status === this.selectedStatus())
          .map((mds: any) => mds.fantacalcio_id)
      : undefined;

    const stimaAsta = this.stimaAstaEnabled() && this.numPartecipanti() != null && this.budget() != null;

    this.mantraService.listPlayers({
      ruolo: this.selectedRuolo() || undefined,
      team: this.selectedTeam() || undefined,
      fase7: this.selectedFase7() || this.activeQuickFilter() || undefined,
      search: this.searchInput() || undefined,
      minPrice: this.priceMin() ?? undefined,
      maxPrice: this.priceMax() ?? undefined,
      fantacalcioIds,
      sortBy: this.sortColumn() || undefined,
      sortDir: this.sortDirection() || undefined,
      page: this.currentPage(),
      size: this.pageSize,
      stimaAsta,
      numPartecipanti: stimaAsta ? this.numPartecipanti()! : undefined,
      budget: stimaAsta ? this.budget()! : undefined,
      overrideRuolo: stimaAsta ? this.roleOverrides() : undefined,
    }).subscribe({
      next: (res) => {
        this.mantraPlayers.set(res.items);
        this.total.set(res.total);
        this.stimaAstaActive.set(res.stima_asta_attiva ?? false);
        this.budgetPerGruppo.set(res.budget_per_gruppo ?? null);
        this.loading.set(false);
      },
      error: (e) => {
        this.error.set(e?.message ?? 'Failed to load');
        this.loading.set(false);
      },
    });
  }

  private loadStats(): void {
    this.mantraService.getStats().subscribe({
      next: (s) => this.stats.set(s),
      error: () => {},
    });
  }

  private loadMatchdayStatus(): void {
    this.mantraService.getMatchdayStatus().subscribe({
      next: (res) => {
        const map: Record<number, any> = {};
        for (const item of res.items) {
          map[item.fantacalcio_id] = item;
        }
        this.matchdayStatusMap.set(map);
      },
      error: () => {},
    });
  }

  private loadExpertRatings(seasonStart: number): void {
    this.expertRatingsService.getForSeason(seasonStart).subscribe({
      next: (res) => {
        const map: Record<number, ExpertRatingWithFantacalcioId> = {};
        for (const item of res.items) {
          map[item.fantacalcio_id] = item;
        }
        this.expertRatingsMap.set(map);
      },
      error: () => {},
    });
  }

  readonly toggleQuickFilter = (key: string) => {
    this.activeQuickFilter.set(this.activeQuickFilter() === key ? null : key);
    this.selectedFase7.set('');
  };

  readonly onSort = (column: string) => {
    if (this.sortColumn() === column) {
      this.sortDirection.update(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      this.sortColumn.set(column);
      this.sortDirection.set('asc');
    }
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
    this.activeQuickFilter.set(null);
    this.stimaAstaEnabled.set(false);
    this.numPartecipanti.set(null);
    this.budget.set(500);
    this.budgetPerGruppo.set(null);
    this.showRoleOverrides.set(false);
    this.roleOverrides.set(PlayersComponent.defaultRoleOverrides());
  };
}
