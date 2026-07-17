import { Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { forkJoin } from 'rxjs';
import { PlayerQuotation, QuotationRoleAggregate, QuotationStatsResponse } from '../../core/models/quotations.models';
import { QuotationService } from '../../core/services/quotation.service';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { QuotationChartComponent } from './components/quotation-chart/quotation-chart.component';

const ROLE_COLORS: Record<string, string> = {
  GK: '#F59E0B', DEF: '#22C55E', MID: '#3B82F6', FWD: '#EF4444',
};

const MANTRA_ROLES = ['Por', 'Dc', 'Dd', 'Ds', 'B', 'E', 'M', 'C', 'T', 'W', 'A', 'Pc'] as const;

@Component({
  selector: 'app-quotations',
  standalone: true,
  imports: [FormsModule, DecimalPipe, SkeletonComponent, ErrorBoundaryComponent, QuotationChartComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Page header -->
      <div class="flex items-center justify-between border-b px-6 py-3.5"
           style="border-color:var(--color-border)">
        <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Quotations</h1>
        @if (selectedSeason()) {
          <span class="text-xs font-mono" style="color:var(--color-text-secondary)">
            Season {{ selectedSeason() }}/{{ selectedSeason()! + 1 }}
          </span>
        }
      </div>

      <div class="p-6 space-y-6">
        <!-- Summary cards -->
        @if (statsLoading()) {
          <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
            @for (_ of [1,2,3,4]; track $index) { <app-skeleton height="80px" /> }
          </div>
        } @else if (roleSummary().length) {
          <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
            @for (rs of roleSummary(); track rs.role) {
              <div class="card">
                <div class="flex items-center justify-between mb-1">
                  <span class="badge text-white text-xs"
                        [style.background]="roleColor(rs.role)">{{ rs.role }}</span>
                  <span class="text-xs" style="color:var(--color-text-secondary)" title="Media Quotazione d'Acquisto">avg qtA</span>
                </div>
                <p class="text-2xl font-bold tabular-nums" [style.color]="roleColor(rs.role)">
                  {{ rs.avgQtA | number:'1.0-0' }}
                </p>
                <p class="text-xs mt-1" style="color:var(--color-text-secondary)">
                  {{ rs.nPlayers }} players · med {{ rs.medianQtA | number:'1.0-0' }}
                </p>
              </div>
            }
          </div>
        }

        <!-- Season comparison chart -->
        @if (stats()?.bySeasonRole?.length) {
          <div class="card">
            <h2 class="font-semibold mb-3" style="color:var(--color-text-primary)">
              Avg Auction Price by Role &amp; Season
            </h2>
            <app-quotation-chart [data]="stats()!.bySeasonRole" />
          </div>
        }

        <!-- Filters -->
        <div class="card">
          <div class="flex flex-wrap gap-3 items-center mb-4">
            <!-- Season selector -->
            @if (seasons().length) {
              <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                      style="background:var(--color-surface-raised);border-color:var(--color-border);
                             color:var(--color-text-primary)"
                      [ngModel]="selectedSeason()"
                      (ngModelChange)="selectedSeason.set($event); currentPage.set(1)">
                <option [ngValue]="null">All seasons</option>
                @for (s of seasons(); track s) {
                  <option [ngValue]="s">{{ s }}/{{ s + 1 }}</option>
                }
              </select>
            }

            <!-- Role filter chips -->
            <div class="flex gap-2" role="group">
              <button class="rounded-full border px-3 py-1 text-xs font-medium"
                      [style]="selectedRole() === null
                        ? 'background:var(--color-accent);color:#fff;border-color:transparent' + (filterMode() === 'mantra' ? ';opacity:0.5' : '')
                        : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                      (click)="selectedRole.set(null); currentPage.set(1)">All</button>
              @for (role of ['GK','DEF','MID','FWD']; track role) {
                <button class="rounded-full border px-3 py-1 text-xs font-medium"
                        [style]="selectedRole() === role
                          ? 'background:' + roleColor(role) + ';color:#fff;border-color:transparent'
                          : filterMode() === 'mantra'
                            ? 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border);opacity:0.35'
                            : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                        (click)="selectedRole.set(role); selectedRuoloPrimario.set(null); currentPage.set(1)">{{ role }}</button>
              }
            </div>

            <!-- Mantra role filter chips -->
            <div class="flex gap-2" role="group">
              <button class="rounded-full border px-3 py-1 text-xs font-medium"
                      [style]="selectedRuoloPrimario() === null
                        ? 'background:var(--color-accent);color:#fff;border-color:transparent' + (filterMode() === 'classic' ? ';opacity:0.5' : '')
                        : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                      (click)="selectedRuoloPrimario.set(null); currentPage.set(1)"
                      title="Filter by Mantra role">Mantra</button>
              @for (role of mantraRoles; track $index) {
                <button class="rounded-full border px-3 py-1 text-xs font-medium"
                        [style]="selectedRuoloPrimario() === role
                          ? 'background:var(--color-accent);color:#fff;border-color:transparent'
                          : filterMode() === 'classic'
                            ? 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border);opacity:0.35'
                            : 'background:var(--color-surface);color:var(--color-text-secondary);border-color:var(--color-border)'"
                        (click)="selectedRuoloPrimario.set(role); selectedRole.set(null); currentPage.set(1)">{{ role }}</button>
              }
            </div>

            <!-- Search -->
            <input class="rounded-lg border px-3 py-1.5 text-sm outline-none ml-auto"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);
                          color:var(--color-text-primary);width:200px"
                   placeholder="Search player…"
                   [ngModel]="searchInput()"
                   (ngModelChange)="searchInput.set($event)" />
          </div>

          <!-- Table -->
          @if (tableLoading()) {
            <div class="space-y-2">
              @for (_ of [1,2,3,4,5]; track $index) { <app-skeleton height="40px" /> }
            </div>
          } @else if (tableError()) {
            <app-error-boundary [message]="tableError()!" />
          } @else {
            <div class="overflow-x-auto">
              <table class="w-full text-sm" style="border-collapse:collapse">
                <thead>
                  <tr class="border-b text-xs font-medium uppercase tracking-wide"
                      style="border-color:var(--color-border);color:var(--color-text-secondary)">
                    <th class="px-3 py-2 text-left">Player</th>
                    <th class="px-3 py-2 text-left">Team</th>
                    <th class="px-3 py-2 text-center">Role</th>
                    <th class="px-3 py-2 text-center" title="Ruolo primario Mantra">Mantra</th>
                    <th class="px-3 py-2 text-right" title="Quotazione d'Acquisto — prezzo d'asta in crediti">qtA</th>
                    <th class="px-3 py-2 text-right" title="Quotazione Iniziale — prezzo di partenza">qtI</th>
                    <th class="px-3 py-2 text-right" title="Fantacalcio Voto Medio — media voti del giocatore">FVM</th>
                    <th class="px-3 py-2 text-right" title="Differenza qtA - qtI (verde = plusvalenza, rosso = minusvalenza)">Diff</th>
                  </tr>
                </thead>
                <tbody>
                  @for (q of filteredItems(); track q.id) {
                    <tr class="border-b transition-colors"
                        style="border-color:var(--color-border)"
                        (mouseenter)="hoverId = q.id"
                        (mouseleave)="hoverId = null"
                        [style.backgroundColor]="hoverId === q.id ? 'var(--color-surface)' : 'transparent'">
                      <td class="px-3 py-2.5 font-medium" style="color:var(--color-text-primary)">
                        {{ q.playerName }}
                      </td>
                      <td class="px-3 py-2.5 text-xs" style="color:var(--color-text-secondary)">{{ q.team }}</td>
                      <td class="px-3 py-2.5 text-center">
                        <span class="badge text-white text-xs" [style.background]="roleColor(q.role)">
                          {{ q.role }}
                        </span>
                      </td>
                      <td class="px-3 py-2.5 text-center text-xs" style="color:var(--color-text-secondary)">
                        {{ q.ruoloPrimario ?? '—' }}
                      </td>
                      <td class="px-3 py-2.5 text-right font-mono font-semibold"
                          style="color:var(--color-accent)">{{ q.qtA }}</td>
                      <td class="px-3 py-2.5 text-right font-mono text-xs"
                          style="color:var(--color-text-secondary)">{{ q.qtI }}</td>
                      <td class="px-3 py-2.5 text-right font-mono text-xs"
                          style="color:var(--color-text-secondary)">{{ q.fvm ?? '—' }}</td>
                      <td class="px-3 py-2.5 text-right font-mono text-xs"
                          [style.color]="q.diffVal > 0 ? '#22C55E' : q.diffVal < 0 ? '#EF4444' : 'var(--color-text-secondary)'">
                        {{ q.diffVal > 0 ? '+' : '' }}{{ q.diffVal }}
                      </td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>

            @if (totalPages() > 1) {
              <div class="mt-4 flex items-center justify-between text-sm"
                   style="color:var(--color-text-secondary)">
                <span>Page {{ currentPage() }} of {{ totalPages() }}</span>
                <div class="flex gap-2">
                  <button class="rounded-lg border px-3 py-1.5 text-xs"
                          style="border-color:var(--color-border)"
                          [disabled]="currentPage() <= 1"
                          (click)="currentPage.update(p => p - 1)">Prev</button>
                  <button class="rounded-lg border px-3 py-1.5 text-xs"
                          style="border-color:var(--color-border)"
                          [disabled]="currentPage() >= totalPages()"
                          (click)="currentPage.update(p => p + 1)">Next</button>
                </div>
              </div>
            }
          }
        </div>
      </div>
    </div>
  `,
})
export class QuotationsComponent {
  private readonly quotService = inject(QuotationService);

  readonly stats = signal<QuotationStatsResponse | null>(null);
  readonly seasons = signal<number[]>([]);
  readonly statsLoading = signal(true);
  readonly items = signal<PlayerQuotation[]>([]);
  readonly total = signal(0);
  readonly tableLoading = signal(false);
  readonly tableError = signal<string | null>(null);
  readonly selectedSeason = signal<number | null>(null);
  readonly selectedRole = signal<string | null>(null);
  readonly selectedRuoloPrimario = signal<string | null>(null);
  readonly selectedRuoloMantra = signal<string | null>(null);
  readonly searchInput = signal('');
  readonly search = signal('');
  readonly currentPage = signal(1);

  hoverId: number | null = null;
  readonly mantraRoles = MANTRA_ROLES;

  readonly filterMode = computed(() => {
    if (this.selectedRole() !== null) return 'classic';
    if (this.selectedRuoloPrimario() !== null) return 'mantra';
    return null;
  });

  readonly totalPages = computed(() => Math.ceil(this.total() / 50));

  readonly roleSummary = computed((): QuotationRoleAggregate[] => {
    const s = this.stats();
    if (!s) return [];
    const season = this.selectedSeason() ?? s.seasons[0];
    return ['GK', 'DEF', 'MID', 'FWD'].map(role =>
      s.bySeasonRole.find(r => r.seasonStart === season && r.role === role)!
    ).filter(Boolean);
  });

  readonly filteredItems = computed(() => {
    const q = this.search().toLowerCase();
    if (!q) return this.items();
    return this.items().filter(i => i.playerName.toLowerCase().includes(q));
  });

  roleColor(role: string): string { return ROLE_COLORS[role] ?? '#8892AA'; }

  constructor() {
    // Load stats + seasons in parallel on init
    forkJoin({
      stats: this.quotService.getStats(),
      seasons: this.quotService.getSeasons(),
    }).subscribe({
      next: ({ stats, seasons }) => {
        this.stats.set(stats);
        this.seasons.set(seasons);
        this.statsLoading.set(false);
        // Default to latest season
        if (seasons.length) this.selectedSeason.set(seasons[0]);
      },
      error: () => this.statsLoading.set(false),
    });

    // Debounce search
    effect((onCleanup) => {
      const val = this.searchInput();
      const tid = setTimeout(() => this.search.set(val), 300);
      onCleanup(() => clearTimeout(tid));
    });

    // Load table on filter/page change
    effect(() => {
      const season = this.selectedSeason();
      const role = this.selectedRole();
      const ruoloPrimario = this.selectedRuoloPrimario();
      const ruoloMantra = this.selectedRuoloMantra();
      const page = this.currentPage();
      this.tableLoading.set(true);
      this.tableError.set(null);
      this.quotService.getQuotations({
        seasonStart: season ?? undefined,
        role: role ?? undefined,
        ruoloPrimario: ruoloPrimario ?? undefined,
        ruoloMantra: ruoloMantra ?? undefined,
        page,
        size: 50,
      }).subscribe({
        next: res => {
          this.items.set(res.items);
          this.total.set(res.total);
          this.tableLoading.set(false);
        },
        error: () => {
          this.tableError.set('Could not load quotations.');
          this.tableLoading.set(false);
        },
      });
    });
  }
}
