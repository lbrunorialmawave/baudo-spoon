import { Component, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatchStat, Season, League } from '../../core/models/stats.models';
import { MatchService } from '../../core/services/match.service';
import { LeagueService } from '../../core/services/league.service';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { MatchDetailComponent } from './components/match-detail/match-detail.component';

@Component({
  selector: 'app-matches',
  standalone: true,
  imports: [FormsModule, SkeletonComponent, ErrorBoundaryComponent, MatchDetailComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Page header -->
      <div class="flex flex-col gap-1 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">Matches</h1>
        @if (total()) {
          <span class="text-xs" style="color:var(--color-text-secondary)">{{ total() }} results</span>
        }
      </div>

      <!-- Filters -->
      <div class="grid grid-cols-1 gap-2 border-b px-4 py-3 sm:flex sm:flex-wrap sm:items-center sm:gap-3 sm:px-6"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <input class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-45"
               style="background:var(--color-surface-raised);border-color:var(--color-border);
                      color:var(--color-text-primary)"
               placeholder="Search team…"
               [ngModel]="teamFilter()"
               (ngModelChange)="teamFilter.set($event); currentPage.set(1)" />

        @if (leagues().length) {
          <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-auto"
                  style="background:var(--color-surface-raised);border-color:var(--color-border);
                         color:var(--color-text-primary)"
                  [ngModel]="leagueFilter()"
                  (ngModelChange)="leagueFilter.set($event); currentPage.set(1)">
            <option value="">All leagues</option>
            @for (l of leagues(); track l.id) {
              <option [value]="l.name">{{ l.name }}</option>
            }
          </select>
        }

        @if (seasons().length) {
          <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-auto"
                  style="background:var(--color-surface-raised);border-color:var(--color-border);
                         color:var(--color-text-primary)"
                  [ngModel]="seasonFilter()"
                  (ngModelChange)="seasonFilter.set(+$event || null); currentPage.set(1)">
            <option [ngValue]="null">All seasons</option>
            @for (s of seasons(); track s.id) {
              <option [ngValue]="s.id">{{ s.season_label }}</option>
            }
          </select>
        }
      </div>

      <!-- Content -->
      <div class="p-4 sm:p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else if (loading()) {
          <div class="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            @for (_ of skeletonRows; track $index) { <app-skeleton height="96px" /> }
          </div>
        } @else if (!items().length) {
          <p class="text-sm" style="color:var(--color-text-secondary)">No matches found.</p>
        } @else {
          <div class="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            @for (m of items(); track m.id) {
              <div class="card cursor-pointer transition-opacity hover:opacity-90"
                   (click)="selectedMatch.set(m)">
                <div class="flex items-start justify-between gap-2">
                  <div class="min-w-0">
                    <p class="truncate font-medium text-sm" style="color:var(--color-text-primary)">
                      {{ m.match_name }}
                    </p>
                    <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">
                      {{ m.match_date ?? '—' }}
                      @if (m.round_num) { · Round {{ m.round_num }} }
                    </p>
                  </div>
                  @if (m.score) {
                    <span class="shrink-0 font-mono font-bold text-sm"
                          style="color:var(--color-text-primary)">{{ m.score }}</span>
                  }
                </div>
                <div class="mt-2 flex items-center gap-3 text-xs"
                     style="color:var(--color-text-secondary)">
                  <span>{{ m.team }}</span>
                  @if (m.points !== null) {
                    <span class="font-semibold" style="color:var(--color-accent)">
                      {{ m.points }} pts
                    </span>
                  }
                  @if (m.season.season_label) {
                    <span class="ml-auto">{{ m.season.season_label }}</span>
                  }
                </div>
              </div>
            }
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

    @if (selectedMatch(); as m) {
      <app-match-detail [match]="m" (closed)="selectedMatch.set(null)" />
    }
  `,
})
export class MatchesComponent {
  private readonly matchService = inject(MatchService);
  private readonly leagueService = inject(LeagueService);

  readonly leagues = signal<League[]>([]);
  readonly seasons = signal<Season[]>([]);
  readonly items = signal<MatchStat[]>([]);
  readonly total = signal(0);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly currentPage = signal(1);
  readonly selectedMatch = signal<MatchStat | null>(null);
  readonly teamFilter = signal('');
  readonly leagueFilter = signal('');
  readonly seasonFilter = signal<number | null>(null);

  readonly skeletonRows = Array.from({ length: 6 });
  readonly totalPages = () => Math.ceil(this.total() / 20);

  constructor() {
    // Load reference data
    this.leagueService.getLeagues().subscribe({ next: l => this.leagues.set(l), error: () => {} });
    this.leagueService.getSeasons().subscribe({ next: s => this.seasons.set(s), error: () => {} });

    effect(() => {
      const team = this.teamFilter();
      const league = this.leagueFilter();
      const season = this.seasonFilter();
      const page = this.currentPage();
      this.loading.set(true);
      this.error.set(null);
      this.matchService.getMatches({
        team: team || undefined,
        league: league || undefined,
        season: season ?? undefined,
        page,
        size: 20,
      }).subscribe({
        next: res => {
          this.items.set(res.items);
          this.total.set(res.total);
          this.loading.set(false);
        },
        error: () => {
          this.error.set('Could not load matches. Check API connection.');
          this.loading.set(false);
        },
      });
    });
  }
}
