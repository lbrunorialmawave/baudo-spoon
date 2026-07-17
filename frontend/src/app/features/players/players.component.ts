import { Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { StatsService } from '../../core/services/stats.service';
import { MantraService } from '../../core/services/mantra.service';
import { FASE7_LABELS, MANTRA_ROLES, MantraPlayer } from '../../core/models/mantra.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { PlayerTableComponent } from './components/player-table/player-table.component';
import { PlayerDrawerComponent } from './components/player-drawer/player-drawer.component';

@Component({
  selector: 'app-players',
  standalone: true,
  imports: [FormsModule, DecimalPipe, ErrorBoundaryComponent, PlayerTableComponent, PlayerDrawerComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <div class="flex items-center justify-between border-b px-6 py-3.5"
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

      <div class="flex flex-wrap items-center gap-3 border-b px-6 py-3"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <input class="rounded-lg border px-3 py-1.5 text-sm outline-none"
               style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary);width:200px"
               placeholder="Search player" [ngModel]="searchInput()" (ngModelChange)="searchInput.set($event)" />

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedRuolo()" (ngModelChange)="selectedRuolo.set($event)">
          <option value="">All roles</option>
          @for (r of MANTRA_ROLES; track r) { <option [value]="r">{{ r }}</option> }
        </select>

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedFase7()" (ngModelChange)="selectedFase7.set($event)">
          <option value="">All classifications</option>
          @for (key of FASE7_KEYS; track key) {
            <option [value]="key">{{ FASE7_LABELS[key].icon }} {{ FASE7_LABELS[key].label }}</option>
          }
        </select>

        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                [ngModel]="selectedStatus()" (ngModelChange)="selectedStatus.set($event)">
          <option value="">All statuses</option>
          <option value="starter">🟢 Titolare</option>
          <option value="bench">⚪ Panchina</option>
          <option value="injured">🔴 Infortunato</option>
          <option value="suspended">🔴 Squalificato</option>
          <option value="doubtful">🟡 In dubbio</option>
        </select>

        <div class="flex gap-1.5 ml-auto">
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
      </div>

      @if (stats(); as s) {
        <div class="flex gap-4 px-6 py-2 border-b text-xs"
             style="border-color:var(--color-border);color:var(--color-text-secondary);background:var(--color-surface)">
          <span title="Media Fantacalcio Punti di tutti i giocatori">Avg FP: <strong style="color:var(--color-accent)">{{ s.avg_fp_mantra | number:'1.1-1' }}</strong></span>
          <span title="Media Voto Ricevuto di tutti i giocatori (scala 0-100)">Avg VR: <strong>{{ s.avg_vr | number:'1.0-0' }}</strong></span>
          <span class="ml-auto">{{ s.total_players }} scored</span>
        </div>
      }

      <div class="p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else {
          <div class="card p-0 overflow-hidden">
            <app-player-table
              [items]="filteredItems()"
              [loading]="loading()"
              [page]="currentPage()"
              [pageSize]="pageSize"
              [mantraMap]="mantraMap()"
              [matchdayStatus]="matchdayStatusMap()"
              [sortColumn]="sortColumn()"
              [sortDirection]="sortDirection()"
              (sortChanged)="onSort($any($event))"
              (playerSelected)="selectedPlayer.set($event)" />
          </div>
          @let displayPages = displayPagesComputed();
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

  readonly MANTRA_ROLES = MANTRA_ROLES;
  readonly FASE7_LABELS = FASE7_LABELS;
  readonly FASE7_KEYS = Object.keys(FASE7_LABELS);

  readonly FASE7_TOOLTIPS: Record<string, string> = {
    TOP: '🏆 TOP — Giocatore d\'élite: FP alto e VR bilanciato. Investimento sicuro.',
    AFFARE: '💎 AFFARE — Sottovalutato dal mercato: FP alto, prezzo basso. Ottimo rapporto Q/P.',
    SCOMMESSA: '🔄 SCOMMESSA — Potenziale inespresso: FP basso ma VR alto. Può esplodere.',
    CERTEZZA: '✅ CERTEZZA — Rendimento stabile e affidabile. Poche sorprese.',
  };

  readonly quickFilters = [
    { key: 'TOP', label: 'TOP', icon: '\u{1F3C6}' },
    { key: 'AFFARE', label: 'Affari', icon: '\u{1F48E}' },
    { key: 'CERTEZZA', label: 'Certezze', icon: '\u2705' },
    { key: 'SCOMMESSA', label: 'Scommesse', icon: '\u{1F504}' },
  ];

  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly mantraPlayers = signal<MantraPlayer[]>([]);
  readonly stats = signal<{ avg_fp_mantra: number; avg_vr: number; total_players: number } | null>(null);
  readonly total = signal(0);
  readonly currentPage = signal(1);
  readonly pageSize = 50;
  readonly selectedRuolo = signal('');
  readonly selectedFase7 = signal('');
  readonly selectedStatus = signal('');
  readonly searchInput = signal('');
  readonly activeQuickFilter = signal<string | null>(null);
  readonly selectedPlayer = signal<any | null>(null);
  readonly displayItems = signal<any[]>([]);
  readonly matchdayStatusMap = signal<Record<number, any>>({});
  readonly sortColumn = signal<string>('');
  readonly sortDirection = signal<'asc' | 'desc'>('asc');

  readonly mantraMap = computed(() => {
    const map: Record<number, any> = {};
    for (const mp of this.mantraPlayers()) {
      map[mp.fantacalcio_id] = mp;
    }
    return map;
  });

  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize)));

  readonly displayPagesComputed = computed(() => {
    const total = this.selectedStatus() ? this.filteredItems().length : this.total();
    return Math.max(1, Math.ceil(total / this.pageSize));
  });

  readonly filteredItems = computed(() => {
    const items = this.mantraPlayers();
    const statusFilter = this.selectedStatus();
    if (!statusFilter) {
      return items;
    }
    const statusMap = this.matchdayStatusMap();
    return items.filter(item => {
      const mds = statusMap[item.fantacalcio_id];
      return mds && mds.status === statusFilter;
    });
  });

  constructor() {
    effect(() => {
      const r = this.selectedRuolo();
      const f7 = this.selectedFase7();
      const sq = this.activeQuickFilter();
      const search = this.searchInput();
      const status = this.selectedStatus();
      const page = this.currentPage();
      const sortCol = this.sortColumn();
      const sortDir = this.sortDirection();
      this.loadData();
    });
    this.loadStats();
    this.loadMatchdayStatus();
  }

  private loadData(): void {
    this.loading.set(true);
    this.error.set(null);
    this.mantraService.listPlayers({
      ruolo: this.selectedRuolo() || undefined,
      fase7: this.selectedFase7() || this.activeQuickFilter() || undefined,
      search: this.searchInput() || undefined,
      sortBy: this.sortColumn() || undefined,
      sortDir: this.sortDirection() || undefined,
      page: this.currentPage(),
      size: this.pageSize,
    }).subscribe({
      next: (res) => {
        this.mantraPlayers.set(res.items);
        this.total.set(res.total);
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

  readonly toggleQuickFilter = (key: string) => {
    this.activeQuickFilter.set(this.activeQuickFilter() === key ? null : key);
    this.selectedFase7.set('');
    this.currentPage.set(1);
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
    this.selectedFase7.set('');
    this.searchInput.set('');
    this.activeQuickFilter.set(null);
    this.currentPage.set(1);
  };
}
