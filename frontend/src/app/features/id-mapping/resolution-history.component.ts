import { Component, computed, inject, signal } from '@angular/core';
import { DecimalPipe, DatePipe } from '@angular/common';
import { IdMappingService } from '../../core/services/id-mapping.service';
import {
  ManualResolution,
  ManualResolutionListResponse,
  ManualResolutionStatsResponse,
} from '../../core/models/quotations.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

@Component({
  selector: 'app-resolution-history',
  standalone: true,
  imports: [DecimalPipe, DatePipe, ErrorBoundaryComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b px-4 py-3 sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <div class="flex items-center gap-3 min-w-0">
          <h1 class="text-base font-semibold truncate" style="color:var(--color-text-primary)">
            Resolution History
          </h1>
          <span class="rounded-full px-2 py-0.5 text-xs font-medium shrink-0"
                style="background:var(--color-surface-raised);color:var(--color-text-secondary)">
            Permanent manual overrides
          </span>
        </div>
        @if (total()) {
          <span class="text-xs" style="color:var(--color-text-secondary)">
            {{ total() }} resolutions · {{ uniquePlayers() }} unique players
          </span>
        }
      </div>

      <!-- Stats cards -->
      @if (statsLoading()) {
        <div class="grid grid-cols-2 gap-3 p-4 sm:p-6 lg:grid-cols-4">
          @for (_ of [1,2,3,4]; track $index) {
            <div class="card h-20 animate-pulse" style="background:var(--color-surface)"></div>
          }
        </div>
      } @else {
        <div class="grid grid-cols-2 gap-3 p-4 sm:p-6 lg:grid-cols-4">
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Total Resolutions</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:var(--color-accent)">
              {{ stats()?.total ?? 0 }}
            </p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Unique Players</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:#8B5CF6">{{ stats()?.uniquePlayers ?? 0 }}</p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Seasons Covered</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:var(--color-text-primary)">
              {{ seasonCount() }}
            </p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Avg per Season</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:#22C55E">
              {{ avgPerSeason() | number:'1.0-0' }}
            </p>
          </div>
        </div>
      }

      <!-- Filters -->
      <div class="flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-2 sm:gap-3 border-b px-4 py-3 sm:px-6"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <!-- Season -->
        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-auto"
                style="background:var(--color-surface-raised);border-color:var(--color-border);
                       color:var(--color-text-primary)"
                (change)="onSeasonChange($event)">
          <option value="" [selected]="selectedSeason() === null">All seasons</option>
          @for (s of seasons(); track s) {
            <option [value]="s" [selected]="selectedSeason() === s">{{ s }}/{{ s + 1 }}</option>
          }
        </select>

        <!-- Search -->
        <input type="text" placeholder="Search player name..."
               class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-48"
               style="background:var(--color-surface-raised);border-color:var(--color-border);
                      color:var(--color-text-primary)"
               [value]="searchQuery()"
               (input)="onSearch($event)" />

        @if (total()) {
          <span class="text-xs sm:ml-auto" style="color:var(--color-text-secondary)">
            Page {{ currentPage() }} of {{ totalPages() }}
          </span>
        }
      </div>

      <!-- Table -->
      <div class="p-4 sm:p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else {
          <div class="card p-0 overflow-hidden">
            <div class="overflow-x-auto" style="-webkit-overflow-scrolling:touch">
              <table class="w-full text-sm" style="min-width:640px">
                <thead>
                  <tr style="color:var(--color-text-secondary);border-color:var(--color-border)">
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs">Fantacalcio</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden md:table-cell">Season</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">Team</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs">Role</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">FotMob Name</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">FotMob ID</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">Note</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs">Resolved</th>
                    <th class="text-right px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs">Action</th>
                  </tr>
                </thead>
                <tbody>
                  @if (loading()) {
                    @for (_ of [1,2,3,4,5]; track $index) {
                      <tr><td colspan="9" class="px-3 py-3 sm:px-4"><div class="h-4 animate-pulse rounded" style="background:var(--color-surface)"></div></td></tr>
                    }
                  } @else if (items().length === 0) {
                    <tr><td colspan="9" class="px-3 py-8 sm:px-4 text-center text-sm" style="color:var(--color-text-secondary)">
                      No manual resolutions found.
                    </td></tr>
                  } @else {
                    @for (item of items(); track item.id) {
                      <tr class="border-t" style="border-color:var(--color-border)">
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 font-medium" style="color:var(--color-text-primary)">
                          <div class="min-w-0">
                            <div class="truncate">{{ item.nameFantacalcio }}</div>
                            <div class="md:hidden text-xs" style="color:var(--color-text-secondary)">
                              {{ item.seasonStart }}/{{ item.seasonStart + 1 }}
                              @if (item.teamFantacalcio) { · {{ item.teamFantacalcio }} }
                            </div>
                          </div>
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 font-mono text-xs hidden md:table-cell" style="color:var(--color-text-secondary)">
                          {{ item.seasonStart }}/{{ item.seasonStart + 1 }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                          {{ item.teamFantacalcio ?? '—' }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5">
                          <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white whitespace-nowrap"
                                [style.background]="roleColor(item.canonicalRole ?? '')">
                            {{ item.canonicalRole ?? '—' }}
                          </span>
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                          {{ item.nameFotmob ?? '—' }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 font-mono text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                          {{ item.playerFotmobId }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-xs hidden lg:table-cell" style="color:var(--color-text-secondary);max-width:200px">
                          <span class="truncate block">{{ item.note ?? '—' }}</span>
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-xs whitespace-nowrap" style="color:var(--color-text-secondary)">
                          {{ item.createdAt | date:'dd/MM/yy' }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-right">
                          <button class="rounded-lg border px-3 py-1 text-xs font-medium whitespace-nowrap"
                                  style="border-color:#EF4444;color:#EF4444"
                                  (click)="confirmDelete(item)">
                            Delete
                          </button>
                        </td>
                      </tr>
                    }
                  }
                </tbody>
              </table>
            </div>
          </div>

          <!-- Pagination -->
          @if (totalPages() > 1) {
            <div class="mt-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-sm"
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

    <!-- Delete confirmation modal -->
    @if (deleting()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4"
           style="background:rgba(0,0,0,0.4)"
           (click)="cancelDelete()">
        <div class="rounded-xl shadow-xl w-full max-w-sm mx-4 overflow-hidden"
             style="background:var(--color-bg);border:1px solid var(--color-border)"
             (click)="$event.stopPropagation()">
          <div class="px-4 py-3 sm:px-5 sm:py-4 border-b" style="border-color:var(--color-border)">
            <h2 class="text-sm font-semibold" style="color:var(--color-text-primary)">
              Delete Resolution?
            </h2>
          </div>
          <div class="px-4 py-3 sm:px-5 sm:py-4 text-sm" style="color:var(--color-text-secondary)">
            <p>Are you sure you want to remove the permanent resolution for</p>
            <p class="font-semibold mt-1" style="color:var(--color-text-primary)">
              {{ deleteTarget()?.nameFantacalcio }}
            </p>
            <p class="mt-1">This will not affect the current player_id_map, but the resolution will no longer be applied automatically in future mapping runs.</p>
          </div>
          <div class="flex justify-end gap-2 px-4 py-3 sm:px-5 sm:py-4 border-t" style="border-color:var(--color-border)">
            <button class="rounded-lg border px-3 py-1.5 text-xs"
                    style="border-color:var(--color-border)"
                    (click)="cancelDelete()">Cancel</button>
            <button class="rounded-lg px-3 py-1.5 text-xs text-white font-medium"
                    style="background:#EF4444"
                    (click)="doDelete()">Delete</button>
          </div>
        </div>
      </div>
    }
  `,
})
export class ResolutionHistoryComponent {
  private readonly service = inject(IdMappingService);

  // ── State ──────────────────────────────────────────────────────────────
  readonly items = signal<ManualResolution[]>([]);
  readonly total = signal(0);
  readonly currentPage = signal(1);
  readonly pageSize = signal(50);
  readonly loading = signal(false);
  readonly statsLoading = signal(false);
  readonly error = signal<string | null>(null);
  readonly stats = signal<ManualResolutionStatsResponse | null>(null);
  readonly selectedSeason = signal<number | null>(null);
  readonly searchQuery = signal('');

  // ── Delete modal state ─────────────────────────────────────────────────
  readonly deleting = signal(false);
  readonly deleteTarget = signal<ManualResolution | null>(null);

  // ── Computed ───────────────────────────────────────────────────────────
  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize())));
  readonly uniquePlayers = computed(() => this.stats()?.uniquePlayers ?? 0);
  readonly seasonCount = computed(() => Object.keys(this.stats()?.bySeason ?? {}).length);
  readonly avgPerSeason = computed(() => {
    const s = this.stats();
    if (!s || s.total === 0) return 0;
    const n = Object.keys(s.bySeason).length;
    return n > 0 ? s.total / n : 0;
  });
  readonly seasons = computed(() => {
    const s = this.stats();
    return s ? Object.keys(s.bySeason).map(Number).sort((a, b) => b - a) : [];
  });

  // ── Lifecycle ──────────────────────────────────────────────────────────
  constructor() {
    this.loadStats();
    this.loadData();
  }

  // ── Data loading ───────────────────────────────────────────────────────
  private loadData(): void {
    this.loading.set(true);
    this.error.set(null);
    this.service.listResolutions({
      seasonStart: this.selectedSeason() ?? undefined,
      search: this.searchQuery() || undefined,
      page: this.currentPage(),
      size: this.pageSize(),
    }).subscribe({
      next: (res: ManualResolutionListResponse) => {
        this.items.set(res.items);
        this.total.set(res.total);
        this.loading.set(false);
      },
      error: (err: unknown) => {
        this.loading.set(false);
        this.error.set(err instanceof Error ? err.message : 'Failed to load resolutions');
      },
    });
  }

  private loadStats(): void {
    this.statsLoading.set(true);
    this.service.getResolutionStats().subscribe({
      next: (res: ManualResolutionStatsResponse) => {
        this.stats.set(res);
        this.statsLoading.set(false);
      },
      error: () => {
        this.statsLoading.set(false);
      },
    });
  }

  // ── Filters ────────────────────────────────────────────────────────────
  onSeasonChange(event: Event): void {
    const val = (event.target as HTMLSelectElement).value;
    this.selectedSeason.set(val ? Number(val) : null);
    this.currentPage.set(1);
    this.loadData();
  }

  onSearch(event: Event): void {
    const val = (event.target as HTMLInputElement).value;
    this.searchQuery.set(val);
    this.currentPage.set(1);
    // Debounce: reload after 300ms of no typing
    clearTimeout((this as any)._searchTimer);
    (this as any)._searchTimer = setTimeout(() => this.loadData(), 300);
  }

  // ── Delete flow ────────────────────────────────────────────────────────
  confirmDelete(item: ManualResolution): void {
    this.deleteTarget.set(item);
    this.deleting.set(true);
  }

  cancelDelete(): void {
    this.deleting.set(false);
    this.deleteTarget.set(null);
  }

  doDelete(): void {
    const target = this.deleteTarget();
    if (!target) return;
    this.service.deleteResolution(target.id).subscribe({
      next: () => {
        this.deleting.set(false);
        this.deleteTarget.set(null);
        // Reload current page
        this.loadData();
        this.loadStats();
      },
      error: (err: unknown) => {
        this.deleting.set(false);
        this.error.set(err instanceof Error ? err.message : 'Failed to delete resolution');
      },
    });
  }

  // ── Helpers ────────────────────────────────────────────────────────────
  roleColor(role: string): string {
    const colors: Record<string, string> = {
      GK: '#22C55E', DEF: '#3B82F6', MID: '#F59E0B', FWD: '#EF4444',
    };
    return colors[role] ?? '#6B7280';
  }
}
