import { Component, computed, effect, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { IdMappingService } from '../../core/services/id-mapping.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  IdMappingListResponse,
  IdMappingStatsResponse,
  PlayerIdMapping,
  UpdateIdMappingRequest,
} from '../../core/models/quotations.models';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

type SortField = 'seasonStart' | 'matchMethod' | 'confidence' | 'nameFantacalcio';
type SortDir = 'asc' | 'desc';

const METHOD_LABELS: Record<string, string> = {
  exact_name_team: 'Exact (name+team)',
  exact_name_team_role_season: 'Exact (name+team+role+season)',
  exact_name_role: 'Exact (name+role)',
  exact_relaxed_role: 'Exact (relaxed role)',
  fuzzy_name: 'Fuzzy',
  manual: '👤 Manual',
  unmatched: '❌ Unmatched',
};

const METHOD_COLORS: Record<string, string> = {
  unmatched: '#EF4444',
  fuzzy_name: '#F59E0B',
  exact_relaxed_role: '#22C55E',
  exact_name_team_role_season: '#3B82F6',
  exact_name_team: '#3B82F6',
  exact_name_role: '#3B82F6',
  manual: '#8B5CF6',
};

@Component({
  selector: 'app-id-mapping',
  standalone: true,
  imports: [DecimalPipe, ErrorBoundaryComponent],
  template: `
    <div style="background:var(--color-bg);min-height:100%">
      <!-- Header -->
      <div class="flex items-center justify-between border-b px-6 py-3.5"
           style="border-color:var(--color-border)">
        <div class="flex items-center gap-3">
          <h1 class="text-base font-semibold" style="color:var(--color-text-primary)">
            ID Mapping
          </h1>
          <span class="rounded-full px-2 py-0.5 text-xs font-medium"
                style="background:var(--color-surface-raised);color:var(--color-text-secondary)">
            Fantacalcio ↔ FotMob
          </span>
        </div>
        @if (total()) {
          <span class="text-xs" style="color:var(--color-text-secondary)">
            {{ total() }} rows · {{ matchedCount() }} matched
            ({{ matchRate() | number:'1.0-1' }}%)
          </span>
        }
      </div>

      <!-- Stats cards -->
      @if (statsLoading()) {
        <div class="grid grid-cols-4 gap-3 p-6">
          @for (_ of [1,2,3,4]; track $index) {
            <div class="card h-20 animate-pulse" style="background:var(--color-surface)"></div>
          }
        </div>
      } @else {
        <div class="grid grid-cols-4 gap-3 p-6">
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Match Rate</p>
            <p class="text-2xl font-bold" style="color:var(--color-accent)">
              {{ matchRate() | number:'1.0-1' }}%
            </p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Unmatched</p>
            <p class="text-2xl font-bold" style="color:#EF4444">{{ unmatchedCount() }}</p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Manual</p>
            <p class="text-2xl font-bold" style="color:#8B5CF6">{{ manualCount() }}</p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Total</p>
            <p class="text-2xl font-bold" style="color:var(--color-text-primary)">{{ stats()?.total ?? 0 }}</p>
          </div>
        </div>
      }

      <!-- Filters -->
      <div class="flex flex-wrap items-center gap-3 border-b px-6 py-3"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <!-- Season -->
        @if (seasons().length) {
          <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                  style="background:var(--color-surface-raised);border-color:var(--color-border);
                         color:var(--color-text-primary)"
                  (change)="onSeasonChange($event)">
            <option value="" [selected]="selectedSeason() === null">All seasons</option>
            @for (s of seasons(); track s) {
              <option [value]="s" [selected]="selectedSeason() === s">{{ s }}/{{ s + 1 }}</option>
            }
          </select>
        }

        <!-- Match method -->
        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                style="background:var(--color-surface-raised);border-color:var(--color-border);
                       color:var(--color-text-primary)"
                (change)="onMethodChange($event)">
          <option value="" [selected]="selectedMethod() === ''">All methods</option>
          <option value="unmatched" [selected]="selectedMethod() === 'unmatched'">❌ Unmatched</option>
          <option value="fuzzy_name" [selected]="selectedMethod() === 'fuzzy_name'">Fuzzy</option>
          <option value="exact_relaxed_role" [selected]="selectedMethod() === 'exact_relaxed_role'">Relaxed role</option>
          <option value="exact_name_team_role_season" [selected]="selectedMethod() === 'exact_name_team_role_season'">Exact (season)</option>
          <option value="exact_name_team" [selected]="selectedMethod() === 'exact_name_team'">Exact (name+team)</option>
          <option value="exact_name_role" [selected]="selectedMethod() === 'exact_name_role'">Exact (name+role)</option>
          <option value="manual" [selected]="selectedMethod() === 'manual'">👤 Manual</option>
        </select>

        <!-- Role -->
        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none"
                style="background:var(--color-surface-raised);border-color:var(--color-border);
                       color:var(--color-text-primary)"
                (change)="onRoleChange($event)">
          <option value="" [selected]="selectedRole() === ''">All roles</option>
          <option value="GK" [selected]="selectedRole() === 'GK'">GK</option>
          <option value="DEF" [selected]="selectedRole() === 'DEF'">DEF</option>
          <option value="MID" [selected]="selectedRole() === 'MID'">MID</option>
          <option value="FWD" [selected]="selectedRole() === 'FWD'">FWD</option>
        </select>

        <!-- Quick filter: only unresolved -->
        <label class="flex items-center gap-1.5 text-sm cursor-pointer"
               style="color:var(--color-text-secondary)">
          <input type="checkbox"
                 [checked]="unresolvedOnly()"
                 (change)="onUnresolvedChange($event)" />
          Unresolved only
        </label>

        @if (total()) {
          <span class="text-xs ml-auto" style="color:var(--color-text-secondary)">
            Page {{ currentPage() }} of {{ totalPages() }}
          </span>
        }
      </div>

      <!-- Table -->
      <div class="p-6">
        @if (error()) {
          <app-error-boundary [message]="error()!" />
        } @else {
          <div class="card p-0 overflow-hidden">
            <table class="w-full text-sm">
              <thead>
                <tr style="color:var(--color-text-secondary);border-color:var(--color-border)">
                  <th class="text-left px-4 py-2.5 font-medium text-xs cursor-pointer select-none"
                      (click)="toggleSort('nameFantacalcio')">
                    Fantacalcio @if (sortField() === 'nameFantacalcio') { {{ sortDir() === 'asc' ? '▲' : '▼' }} }
                  </th>
                  <th class="text-left px-4 py-2.5 font-medium text-xs cursor-pointer select-none"
                      (click)="toggleSort('seasonStart')">
                    Season @if (sortField() === 'seasonStart') { {{ sortDir() === 'asc' ? '▲' : '▼' }} }
                  </th>
                  <th class="text-left px-4 py-2.5 font-medium text-xs">Team</th>
                  <th class="text-left px-4 py-2.5 font-medium text-xs">Role</th>
                  <th class="text-left px-4 py-2.5 font-medium text-xs cursor-pointer select-none"
                      (click)="toggleSort('matchMethod')">
                    Method @if (sortField() === 'matchMethod') { {{ sortDir() === 'asc' ? '▲' : '▼' }} }
                  </th>
                  <th class="text-left px-4 py-2.5 font-medium text-xs">FotMob Name</th>
                  <th class="text-left px-4 py-2.5 font-medium text-xs">FotMob ID</th>
                  <th class="text-right px-4 py-2.5 font-medium text-xs">Action</th>
                </tr>
              </thead>
              <tbody>
                @if (loading()) {
                  @for (_ of [1,2,3,4,5]; track $index) {
                    <tr><td colspan="8" class="px-4 py-3"><div class="h-4 animate-pulse rounded" style="background:var(--color-surface)"></div></td></tr>
                  }
                } @else if (sortedItems().length === 0) {
                  <tr><td colspan="8" class="px-4 py-8 text-center text-sm" style="color:var(--color-text-secondary)">
                    No rows found.
                  </td></tr>
                } @else {
                  @for (item of sortedItems(); track item.id) {
                    <tr class="border-t" style="border-color:var(--color-border)">
                      <td class="px-4 py-2.5 font-medium" style="color:var(--color-text-primary)">
                        {{ item.nameFantacalcio }}
                      </td>
                      <td class="px-4 py-2.5 font-mono text-xs" style="color:var(--color-text-secondary)">
                        {{ item.seasonStart }}/{{ item.seasonStart + 1 }}
                      </td>
                      <td class="px-4 py-2.5 text-xs" style="color:var(--color-text-secondary)">
                        {{ item.teamFantacalcio }}
                      </td>
                      <td class="px-4 py-2.5">
                        <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                              [style.background]="roleColor(item.canonicalRole ?? '')">
                          {{ item.canonicalRole ?? '—' }}
                        </span>
                      </td>
                      <td class="px-4 py-2.5">
                        <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                              [style.background]="methodColor(item.matchMethod)">
                          {{ methodLabel(item.matchMethod) }}
                        </span>
                      </td>
                      <td class="px-4 py-2.5 text-xs" style="color:var(--color-text-secondary)">
                        {{ item.nameFotmob ?? '—' }}
                      </td>
                      <td class="px-4 py-2.5 font-mono text-xs" style="color:var(--color-text-secondary)">
                        {{ item.playerFotmobId ?? '—' }}
                      </td>
                      <td class="px-4 py-2.5 text-right">
                        @if (item.matchMethod === 'unmatched' || item.matchMethod === 'fuzzy_name') {
                          <button class="rounded-lg border px-3 py-1 text-xs font-medium"
                                  style="border-color:var(--color-accent);color:var(--color-accent)"
                                  (click)="openResolver(item)">
                            Resolve
                          </button>
                        } @else if (item.matchMethod === 'manual') {
                          <button class="rounded-lg border px-3 py-1 text-xs font-medium"
                                  style="border-color:var(--color-border);color:var(--color-text-secondary)"
                                  (click)="openResolver(item)">
                            Edit
                          </button>
                        } @else {
                          <span class="text-xs" style="color:var(--color-text-secondary)">OK</span>
                        }
                      </td>
                    </tr>
                  }
                }
              </tbody>
            </table>
          </div>

          <!-- Pagination -->
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

    <!-- Resolve modal -->
    @if (resolving()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center"
           style="background:rgba(0,0,0,0.4)"
           (click)="closeResolver()">
        <div class="rounded-xl shadow-xl w-full max-w-lg mx-4 overflow-hidden"
             style="background:var(--color-bg);border:1px solid var(--color-border)"
             (click)="$event.stopPropagation()">
          <!-- Modal header -->
          <div class="flex items-center justify-between px-5 py-4 border-b"
               style="border-color:var(--color-border)">
            <h2 class="text-sm font-semibold" style="color:var(--color-text-primary)">
              Resolve ID Mapping
            </h2>
            <button class="text-lg leading-none" style="color:var(--color-text-secondary)"
                    (click)="closeResolver()">✕</button>
          </div>

          <!-- Modal body -->
          <div class="p-5 space-y-4">
            <!-- Current data (read-only) -->
            <div class="rounded-lg p-3 text-xs space-y-1"
                 style="background:var(--color-surface);color:var(--color-text-secondary)">
              <p><strong style="color:var(--color-text-primary)">Fantacalcio:</strong>
                {{ resolveItem()!.nameFantacalcio }}
                @if (resolveItem()!.teamFantacalcio) { · {{ resolveItem()!.teamFantacalcio }} }
                @if (resolveItem()!.canonicalRole) { · {{ resolveItem()!.canonicalRole }} }
                · {{ resolveItem()!.seasonStart }}/{{ resolveItem()!.seasonStart! + 1 }}
              </p>
              <p><strong style="color:var(--color-text-primary)">Current match:</strong>
                {{ methodLabel(resolveItem()!.matchMethod) }}
                @if (resolveItem()!.playerFotmobId) {
                  · FotMob ID {{ resolveItem()!.playerFotmobId }}
                  @if (resolveItem()!.nameFotmob) { ({{ resolveItem()!.nameFotmob }}) }
                }
              </p>
            </div>

            <!-- FotMob ID input -->
            <div>
              <label class="block text-xs font-medium mb-1"
                     style="color:var(--color-text-secondary)">FotMob Player ID</label>
              <input class="w-full rounded-lg border px-3 py-2 text-sm outline-none"
                     style="background:var(--color-surface-raised);border-color:var(--color-border);
                            color:var(--color-text-primary)"
                     placeholder="e.g. 1234567"
                     type="number"
                     [value]="formFotmobId() ?? ''"
                     (change)="onFotmobIdChange($event)" />
              <p class="mt-1 text-xs" style="color:var(--color-text-secondary)">
                Inserisci il FotMob ID del giocatore. Lascia vuoto o imposta <code>-1</code> per lasciare non associato.
              </p>
            </div>

            <!-- Optional: name fotmob -->
            <div>
              <label class="block text-xs font-medium mb-1"
                     style="color:var(--color-text-secondary)">FotMob Name (opzionale)</label>
              <input class="w-full rounded-lg border px-3 py-2 text-sm outline-none"
                     style="background:var(--color-surface-raised);border-color:var(--color-border);
                            color:var(--color-text-primary)"
                     placeholder="e.g. Lautaro Martinez"
                     [value]="formFotmobName()"
                     (change)="onFotmobNameChange($event)" />
            </div>

            <!-- Optional: role override -->
            <div>
              <label class="block text-xs font-medium mb-1"
                     style="color:var(--color-text-secondary)">Ruolo (opzionale)</label>
              <select class="w-full rounded-lg border px-3 py-2 text-sm outline-none"
                      style="background:var(--color-surface-raised);border-color:var(--color-border);
                             color:var(--color-text-primary)"
                      (change)="onFormRoleChange($event)">
                <option value="" [selected]="formRole() === ''">— Keep current —</option>
                <option value="GK" [selected]="formRole() === 'GK'">GK</option>
                <option value="DEF" [selected]="formRole() === 'DEF'">DEF</option>
                <option value="MID" [selected]="formRole() === 'MID'">MID</option>
                <option value="FWD" [selected]="formRole() === 'FWD'">FWD</option>
              </select>
            </div>

            <!-- Note -->
            <div>
              <label class="block text-xs font-medium mb-1"
                     style="color:var(--color-text-secondary)">Nota (opzionale)</label>
              <input class="w-full rounded-lg border px-3 py-2 text-sm outline-none"
                     style="background:var(--color-surface-raised);border-color:var(--color-border);
                            color:var(--color-text-primary)"
                     placeholder="Motivo della correzione..."
                     [value]="formNote()"
                     (change)="onNoteChange($event)" />
            </div>

            <!-- Error / success messages -->
            @if (saveError()) {
              <div class="rounded-lg px-3 py-2 text-xs text-white" style="background:#EF4444">
                {{ saveError() }}
              </div>
            }
            @if (saveSuccess()) {
              <div class="rounded-lg px-3 py-2 text-xs text-white" style="background:#22C55E">
                ✅ Mapping aggiornato con successo!
              </div>
            }
          </div>

          <!-- Modal footer -->
          <div class="flex justify-end gap-2 px-5 py-4 border-t"
               style="border-color:var(--color-border)">
            <button class="rounded-lg border px-4 py-2 text-sm"
                    style="border-color:var(--color-border);color:var(--color-text-secondary)"
                    (click)="closeResolver()">Cancel</button>
            <button class="rounded-lg px-4 py-2 text-sm text-white font-medium"
                    style="background:var(--color-accent)"
                    [disabled]="saving()"
                    (click)="saveResolution()">
              @if (saving()) {
                Saving…
              } @else {
                Save Mapping
              }
            </button>
          </div>
        </div>
      </div>
    }
  `,
})
export class IdMappingComponent {
  private readonly idMappingService = inject(IdMappingService);
  private readonly quotationService = inject(QuotationService);

  // ── State ────────────────────────────────────────────────────────────────
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly data = signal<IdMappingListResponse | null>(null);
  readonly stats = signal<IdMappingStatsResponse | null>(null);
  readonly statsLoading = signal(false);
  readonly seasons = signal<number[]>([]);

  readonly selectedSeason = signal<number | null>(null);
  readonly selectedMethod = signal<string>('');
  readonly selectedRole = signal<string>('');
  readonly unresolvedOnly = signal(false);
  readonly currentPage = signal(1);
  readonly pageSize = 30;

  // Sorting
  readonly sortField = signal<SortField>('seasonStart');
  readonly sortDir = signal<SortDir>('desc');

  // Resolve modal
  readonly resolving = signal(false);
  readonly resolveItem = signal<PlayerIdMapping | null>(null);
  readonly formFotmobId = signal<number | null>(null);
  readonly formFotmobName = signal<string>('');
  readonly formRole = signal<string>('');
  readonly formNote = signal<string>('');
  readonly saving = signal(false);
  readonly saveError = signal<string | null>(null);
  readonly saveSuccess = signal(false);

  // ── Computed ──────────────────────────────────────────────────────────────
  readonly total = computed(() => this.data()?.total ?? 0);
  readonly totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize)));

  readonly matchedCount = computed(() => this.stats()?.matched ?? 0);
  readonly unmatchedCount = computed(() => this.stats()?.unmatched ?? 0);
  readonly manualCount = computed(() => this.stats()?.byMethod?.['manual'] ?? 0);
  readonly matchRate = computed(() => this.stats()?.matchRate ?? 0);

  readonly items = computed(() => this.data()?.items ?? []);

  readonly sortedItems = computed(() => {
    const items = [...this.items()];
    const field = this.sortField();
    const dir = this.sortDir();
    items.sort((a, b) => {
      let cmp = 0;
      switch (field) {
        case 'seasonStart':      cmp = a.seasonStart - b.seasonStart; break;
        case 'matchMethod':      cmp = a.matchMethod.localeCompare(b.matchMethod); break;
        case 'confidence':       cmp = a.confidence - b.confidence; break;
        case 'nameFantacalcio':  cmp = a.nameFantacalcio.localeCompare(b.nameFantacalcio); break;
      }
      return dir === 'asc' ? cmp : -cmp;
    });
    return items;
  });

  // ── Helpers ──────────────────────────────────────────────────────────────
  readonly roleColor = (role: string): string => {
    const map: Record<string, string> = { GK: '#F59E0B', DEF: '#22C55E', MID: '#3B82F6', FWD: '#EF4444' };
    return map[role] ?? '#6B7280';
  };

  readonly methodColor = (m: string): string => METHOD_COLORS[m] ?? '#6B7280';
  readonly methodLabel = (m: string): string => METHOD_LABELS[m] ?? m;

  readonly toggleSort = (field: SortField) => {
    if (this.sortField() === field) {
      this.sortDir.update(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      this.sortField.set(field);
      this.sortDir.set(field === 'seasonStart' ? 'desc' : 'asc');
    }
  };

  // ── Lifecycle ────────────────────────────────────────────────────────────
  constructor() {
    this.loadSeasons();
    this.loadStats();
    effect(() => {
      // Track filter signals to reactively reload data
      const _season = this.selectedSeason();
      const _method = this.selectedMethod();
      const _role = this.selectedRole();
      const _unresolved = this.unresolvedOnly();
      const _page = this.currentPage();
      console.debug('Filter changed', { _season, _method, _role, _unresolved, _page });
      this.loadData();
    });
  }

  // ── Data loading ─────────────────────────────────────────────────────────
  private loadSeasons(): void {
    this.quotationService.getSeasons().subscribe({
      next: (s) => this.seasons.set(s),
      error: () => {},
    });
  }

  private loadStats(): void {
    this.statsLoading.set(true);
    this.idMappingService.getStats().subscribe({
      next: (s) => this.stats.set(s),
      error: () => {},
      complete: () => this.statsLoading.set(false),
    });
  }

  private loadData(): void {
    this.loading.set(true);
    this.error.set(null);
    this.idMappingService.list({
      seasonStart: this.selectedSeason() ?? undefined,
      matchMethod: this.selectedMethod() || undefined,
      canonicalRole: this.selectedRole() || undefined,
      matchedOnly: false,
      page: this.currentPage(),
      size: this.pageSize,
    }).subscribe({
      next: (d) => this.data.set(d),
      error: (e) => this.error.set(e?.message ?? 'Failed to load ID mappings'),
      complete: () => this.loading.set(false),
    });
  }

  // ── Change handlers (nativi, senza ngModel per evitare strictTemplates) ──
  readonly onSeasonChange = (e: Event) => {
    const v = (e.target as HTMLSelectElement).value;
    this.selectedSeason.set(v ? Number(v) : null);
    this.currentPage.set(1);
  };
  readonly onMethodChange = (e: Event) => {
    this.selectedMethod.set((e.target as HTMLSelectElement).value);
    this.currentPage.set(1);
  };
  readonly onRoleChange = (e: Event) => {
    this.selectedRole.set((e.target as HTMLSelectElement).value);
    this.currentPage.set(1);
  };
  readonly onUnresolvedChange = (e: Event) => {
    this.unresolvedOnly.set((e.target as HTMLInputElement).checked);
    this.currentPage.set(1);
  };
  readonly onFotmobIdChange = (e: Event) => {
    const v = (e.target as HTMLInputElement).value;
    this.formFotmobId.set(v ? Number(v) : null);
  };
  readonly onFotmobNameChange = (e: Event) => {
    this.formFotmobName.set((e.target as HTMLInputElement).value);
  };
  readonly onFormRoleChange = (e: Event) => {
    this.formRole.set((e.target as HTMLSelectElement).value);
  };
  readonly onNoteChange = (e: Event) => {
    this.formNote.set((e.target as HTMLInputElement).value);
  };

  // ── Modal ────────────────────────────────────────────────────────────────
  readonly openResolver = (item: PlayerIdMapping) => {
    this.resolveItem.set(item);
    this.formFotmobId.set(item.playerFotmobId);
    this.formFotmobName.set(item.nameFotmob ?? '');
    this.formRole.set('');
    this.formNote.set('');
    this.saveError.set(null);
    this.saveSuccess.set(false);
    this.resolving.set(true);
  };

  readonly closeResolver = () => {
    this.resolving.set(false);
    this.resolveItem.set(null);
  };

  readonly saveResolution = () => {
    const item = this.resolveItem();
    if (!item) return;

    this.saving.set(true);
    this.saveError.set(null);
    this.saveSuccess.set(false);

    const body: UpdateIdMappingRequest = {};
    const fotmobId = this.formFotmobId();
    body.playerFotmobId = fotmobId != null ? fotmobId : -1;
    if (this.formFotmobName()) body.nameFotmob = this.formFotmobName();
    if (this.formRole()) body.canonicalRole = this.formRole();
    if (this.formNote()) body.note = this.formNote();

    this.idMappingService.update(item.fantacalcioId, item.seasonStart, body).subscribe({
      next: () => {
        this.saveSuccess.set(true);
        this.saving.set(false);
        this.loadData();
        this.loadStats();
        // Auto-close after short delay
        setTimeout(() => this.closeResolver(), 1500);
      },
      error: (e) => {
        this.saveError.set(e?.error?.detail ?? e?.message ?? 'Failed to save');
        this.saving.set(false);
      },
    });
  };
}
