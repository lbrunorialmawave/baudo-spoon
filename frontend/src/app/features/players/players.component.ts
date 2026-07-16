import { Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PlayerSeasonStat } from '../../core/models/stats.models';
import { StatsService } from '../../core/services/stats.service';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { PlayerTableComponent } from './components/player-table/player-table.component';
import { PlayerDrawerComponent } from './components/player-drawer/player-drawer.component';

@Component({
  selector: 'app-players',
  standalone: true,
  imports: [FormsModule, ErrorBoundaryComponent, PlayerTableComponent, PlayerDrawerComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Page header -->
      <div class="flex items-center justify-between border-b px-6 py-3.5"
           style="border-color:var(--color-border)">
        <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Players</h1>
        @if (total()) {
          <span class="text-xs" style="color:var(--color-text-secondary)">{{ total() }} results</span>
        }
      </div>

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-3 border-b px-6 py-3"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <input
          class="rounded-lg border px-3 py-1.5 text-sm outline-none"
          style="background:var(--color-surface-raised);border-color:var(--color-border);
                 color:var(--color-text-primary);width:220px"
          placeholder="Search player or team…"
          [ngModel]="searchInput()"
          (ngModelChange)="onSearchInput($event)" />

        @if (categories().length) {
          <select
            class="rounded-lg border px-3 py-1.5 text-sm outline-none"
            style="background:var(--color-surface-raised);border-color:var(--color-border);
                   color:var(--color-text-primary)"
            [ngModel]="selectedCategory()"
            (ngModelChange)="selectedCategory.set($event); currentPage.set(1)">
            <option value="">All categories</option>
            @for (cat of categories(); track cat) {
              <option [value]="cat">{{ cat }}</option>
            }
          </select>
        }
      </div>

      <!-- Content -->
      <div class="p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else {
          <div class="card p-0 overflow-hidden">
            <app-player-table
              [items]="items()"
              [loading]="loading()"
              (playerSelected)="selectedPlayer.set($event)" />
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

    @if (selectedPlayer(); as p) {
      <app-player-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
  `,
})
export class PlayersComponent {
  private readonly statsService = inject(StatsService);

  readonly categories = signal<string[]>([]);
  readonly items = signal<PlayerSeasonStat[]>([]);
  readonly total = signal(0);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly currentPage = signal(1);
  readonly selectedPlayer = signal<PlayerSeasonStat | null>(null);
  readonly selectedCategory = signal('');

  // Separate display vs debounced search
  readonly searchInput = signal('');
  readonly search = signal('');

  readonly totalPages = computed(() => Math.ceil(this.total() / 50));

  constructor() {
    // Load categories once
    this.statsService.getPlayerCategories().subscribe({
      next: cats => this.categories.set(cats),
      error: () => {},
    });

    // Debounce search input → search signal
    effect((onCleanup) => {
      const val = this.searchInput();
      const tid = setTimeout(() => {
        this.search.set(val);
        this.currentPage.set(1);
      }, 300);
      onCleanup(() => clearTimeout(tid));
    });

    // Reload when search, category, or page changes
    effect(() => {
      const search = this.search();
      const category = this.selectedCategory();
      const page = this.currentPage();
      this.loading.set(true);
      this.error.set(null);
      this.statsService.getPlayerStats({
        player: search || undefined,
        statCategory: category || undefined,
        season : 2025,
        page,
        size: 50,
      }).subscribe({
        next: res => {
          this.items.set(res.items);
          this.total.set(res.total);
          this.loading.set(false);
        },
        error: () => {
          this.error.set('Could not load player stats. Check API connection.');
          this.loading.set(false);
        },
      });
    });
  }

  onSearchInput(val: string): void {
    this.searchInput.set(val);
  }
}
