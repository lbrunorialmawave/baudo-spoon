import { Component, computed, effect, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { IdMappingService } from '../../core/services/id-mapping.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  IdMappingListResponse,
  IdMappingRunResponse,
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
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b px-4 py-3 sm:px-6 sm:py-3.5"
           style="border-color:var(--color-border)">
        <div class="flex items-center gap-3 min-w-0">
          <h1 class="text-base font-semibold truncate" style="color:var(--color-text-primary)">
            ID Mapping
          </h1>
          <span class="rounded-full px-2 py-0.5 text-xs font-medium shrink-0"
                style="background:var(--color-surface-raised);color:var(--color-text-secondary)">
            Fantacalcio ↔ FotMob
          </span>
        </div>
        <div class="flex items-center gap-2">
          @if (total()) {
            <span class="text-xs" style="color:var(--color-text-secondary)">
              {{ total() }} rows · {{ matchedCount() }} matched
              ({{ matchRate() | number:'1.0-1' }}%)
            </span>
          }
          <button class="rounded-lg border px-3 py-1.5 text-xs font-medium hover:opacity-80"
                  style="background:var(--color-surface-raised);color:var(--color-text-primary);border-color:var(--color-border)"
                  (click)="runAutoMapping()">
            Run Auto Mapping
          </button>
          <button class="rounded-lg border px-3 py-1.5 text-xs font-medium hover:opacity-80"
                  style="background:var(--color-surface-raised);color:var(--color-text-primary);border-color:var(--color-border)"
                  (click)="exportIdMap()">
            Export ID Map
          </button>
        </div>
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
            <p class="text-xs" style="color:var(--color-text-secondary)">Match Rate</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:var(--color-accent)">
              {{ matchRate() | number:'1.0-1' }}%
            </p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Unmatched</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:#EF4444">{{ unmatchedCount() }}</p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Manual</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:#8B5CF6">{{ manualCount() }}</p>
          </div>
          <div class="card">
            <p class="text-xs" style="color:var(--color-text-secondary)">Total</p>
            <p class="text-xl sm:text-2xl font-bold" style="color:var(--color-text-primary)">{{ stats()?.total ?? 0 }}</p>
          </div>
        </div>
      }

      <!-- Filters -->
      <div class="flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-2 sm:gap-3 border-b px-4 py-3 sm:px-6"
           style="border-color:var(--color-border);background:var(--color-surface)">
        <!-- Season -->
        @if (seasons().length) {
          <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-auto"
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
        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-auto"
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
        <select class="rounded-lg border px-3 py-1.5 text-sm outline-none w-full sm:w-auto"
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
        <label class="flex items-center gap-1.5 text-sm cursor-pointer shrink-0"
               style="color:var(--color-text-secondary)">
          <input type="checkbox"
                 [checked]="unresolvedOnly()"
                 (change)="onUnresolvedChange($event)" />
          Unresolved only
        </label>

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
            <div class="table-scroll">
              <table class="w-full text-sm">
                <thead>
                  <tr style="color:var(--color-text-secondary);border-color:var(--color-border)">
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs cursor-pointer select-none"
                        (click)="toggleSort('nameFantacalcio')">
                      Fantacalcio @if (sortField() === 'nameFantacalcio') { {{ sortDir() === 'asc' ? '▲' : '▼' }} }
                    </th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs cursor-pointer select-none hidden md:table-cell"
                        (click)="toggleSort('seasonStart')">
                      Season @if (sortField() === 'seasonStart') { {{ sortDir() === 'asc' ? '▲' : '▼' }} }
                    </th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">Team</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs">Role</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">Mantra Roles</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs cursor-pointer select-none"
                        (click)="toggleSort('matchMethod')">
                      Method @if (sortField() === 'matchMethod') { {{ sortDir() === 'asc' ? '▲' : '▼' }} }
                    </th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">FotMob Name</th>
                    <th class="text-left px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs hidden lg:table-cell">FotMob ID</th>
                    <th class="text-right px-3 py-2 sm:px-4 sm:py-2.5 font-medium text-xs">Action</th>
                  </tr>
                </thead>
                <tbody>
                  @if (loading()) {
                    @for (_ of [1,2,3,4,5]; track $index) {
                      <tr><td colspan="9" class="px-3 py-3 sm:px-4"><div class="h-4 animate-pulse rounded" style="background:var(--color-surface)"></div></td></tr>
                    }
                  } @else if (sortedItems().length === 0) {
                    <tr><td colspan="9" class="px-3 py-8 sm:px-4 text-center text-sm" style="color:var(--color-text-secondary)">
                      No rows found.
                    </td></tr>
                  } @else {
                    @for (item of sortedItems(); track item.id) {
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
                          {{ item.teamFantacalcio }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5">
                          <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white whitespace-nowrap"
                                [style.background]="roleColor(item.canonicalRole ?? '')">
                            {{ item.canonicalRole ?? '—' }}
                          </span>
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                          @if (item.ruoloPrimario) {
                            <span class="font-medium" style="color:var(--color-text-primary)">{{ item.ruoloPrimario }}</span>
                            @if (item.ruoliMantra && item.ruoliMantra.length > 1) {
                              <span class="ml-1 opacity-60">{{ '{' }}{{ item.ruoliMantra.join(', ') }}{{ '}' }}</span>
                            }
                          } @else {
                            <span class="italic opacity-50">—</span>
                          }
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5">
                          <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white whitespace-nowrap"
                                [style.background]="methodColor(item.matchMethod)">
                            {{ methodLabel(item.matchMethod) }}
                          </span>
                          @if (showFromHistory(item)) {
                            <span class="rounded-full px-2 py-0.5 text-xs font-medium ml-1 whitespace-nowrap"
                                  style="background:#8B5CF6;color:white">
                              From history
                            </span>
                          }
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                          {{ item.nameFotmob ?? '—' }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 font-mono text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                          {{ item.playerFotmobId ?? '—' }}
                        </td>
                        <td class="px-3 py-2 sm:px-4 sm:py-2.5 text-right">
                          <div class="flex items-center justify-end gap-1">
                            @if (item.matchMethod !== 'exact_name_team' && item.matchMethod !== 'exact_name_team_role_season' && item.matchMethod !== 'exact_name_role' && item.matchMethod !== 'exact_relaxed_role') {
                              <button class="rounded px-1.5 py-0.5 text-xs hover:opacity-70"
                                      style="color:var(--color-text-secondary)"
                                      title="Cerca su FotMob"
                                      (click)="verifyOnFotmob(item)">
                                🔍
                              </button>
                            }
                            @if (item.matchMethod === 'unmatched' || item.matchMethod === 'fuzzy_name') {
                              <button class="rounded-lg border px-3 py-1 text-xs font-medium whitespace-nowrap"
                                      style="border-color:var(--color-accent);color:var(--color-accent)"
                                      (click)="openResolver(item)">
                                Resolve
                              </button>
                            } @else if (item.matchMethod === 'manual') {
                              <button class="rounded-lg border px-3 py-1 text-xs font-medium whitespace-nowrap"
                                      style="border-color:var(--color-border);color:var(--color-text-secondary)"
                                      (click)="openResolver(item)">
                                Edit
                              </button>
                            } @else {
                              <span class="text-xs" style="color:var(--color-text-secondary)">OK</span>
                            }
                          </div>
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

    <!-- Verify FotMob panel -->
    @if (verifyMode()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4"
           style="background:rgba(0,0,0,0.4)"
           (click)="verifyMode.set(false)">
        <div class="rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden"
             style="background:var(--color-bg);border:1px solid var(--color-border)"
             (click)="$event.stopPropagation()">
          <div class="flex items-center justify-between px-4 py-3 sm:px-5 sm:py-4 border-b"
               style="border-color:var(--color-border)">
            <h2 class="text-sm font-semibold" style="color:var(--color-text-primary)">
              Verifica su FotMob
            </h2>
            <button class="text-lg leading-none px-2 py-1 min-h-8 min-w-8"
                    style="color:var(--color-text-secondary)"
                    (click)="verifyMode.set(false)">✕</button>
          </div>
          <div class="p-4 sm:p-5 space-y-3">
            <p class="text-xs" style="color:var(--color-text-secondary)">
              Cerca <strong>{{ verifyItem()?.nameFantacalcio }}</strong> su FotMob,
              apri la sua pagina, copia la URL e incollala qui sotto.
            </p>
            <input class="w-full rounded-lg border px-3 py-2 text-sm outline-none"
                   style="background:var(--color-surface-raised);border-color:var(--color-border);color:var(--color-text-primary)"
                   placeholder="https://www.fotmob.com/players/314605/alberto-moreno"
                   [value]="verifyUrl()"
                   (input)="verifyUrl.set(($event.target as HTMLInputElement).value)" />
            <div class="flex gap-2">
              <button class="rounded-lg border px-4 py-1.5 text-xs font-medium"
                      style="background:var(--color-accent);color:#fff;border-color:transparent"
                      (click)="parseFotmobUrl()">
                Analizza URL
              </button>
              <button class="rounded-lg border px-4 py-1.5 text-xs font-medium"
                      style="background:var(--color-surface-raised);color:var(--color-text-primary);border-color:var(--color-border)"
                      (click)="verifyMode.set(false)">
                Annulla
              </button>
            </div>
            @if (verifyResult(); as r) {
              <div class="rounded-lg border p-3 text-xs space-y-1"
                   style="border-color:#22c55e44;background:#22c55e11;color:#16a34a">
                <p><strong>ID FotMob:</strong> {{ r.id }}</p>
                <p><strong>Nome:</strong> {{ r.name }}</p>
                <button class="mt-2 rounded-lg border px-4 py-1.5 text-xs font-medium"
                        style="background:var(--color-accent);color:#fff;border-color:transparent"
                        (click)="useVerifyResult()">
                  Usa e apri risoluzione
                </button>
              </div>
            }
          </div>
        </div>
      </div>
    }

    <!-- Resolve modal -->
    @if (resolving()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4"
           style="background:rgba(0,0,0,0.4)"
           (click)="closeResolver()">
        <div class="rounded-xl shadow-xl w-full max-w-lg mx-4 overflow-hidden"
             style="background:var(--color-bg);border:1px solid var(--color-border);max-height:90vh;display:flex;flex-direction:column"
             (click)="$event.stopPropagation()">
          <!-- Modal header -->
          <div class="flex items-center justify-between px-4 py-3 sm:px-5 sm:py-4 border-b shrink-0"
               style="border-color:var(--color-border)">
            <h2 class="text-sm font-semibold" style="color:var(--color-text-primary)">
              Resolve ID Mapping
            </h2>
            <button class="text-lg leading-none px-2 py-1 min-h-8 min-w-8"
                    style="color:var(--color-text-secondary)"
                    aria-label="Close modal"
                    (click)="closeResolver()">✕</button>
          </div>

          <!-- Modal body -->
          <div class="p-4 sm:p-5 space-y-4 overflow-y-auto">
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
              @if (resolveItem()!.ruoloPrimario) {
                <p><strong style="color:var(--color-text-primary)">Mantra:</strong>
                  {{ resolveItem()!.ruoloPrimario }}
                  @if ((resolveItem()!.ruoliMantra?.length ?? 0) > 1) {
                    · {{ '{' }}{{ resolveItem()!.ruoliMantra!.join(', ') }}{{ '}' }}
                  }
                </p>
              }
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

            <!-- MANTRA role override -->
            <div>
              <label class="block text-xs font-medium mb-1"
                     style="color:var(--color-text-secondary)">Ruolo Primario Mantra (opzionale)</label>
              <select class="w-full rounded-lg border px-3 py-2 text-sm outline-none"
                      style="background:var(--color-surface-raised);border-color:var(--color-border);
                             color:var(--color-text-primary)"
                      (change)="onFormMantraRoleChange($event)">
                <option value="" [selected]="formMantraRole() === ''">— Keep current —</option>
                <option value="Por" [selected]="formMantraRole() === 'Por'">Por</option>
                <option value="Dc" [selected]="formMantraRole() === 'Dc'">Dc</option>
                <option value="Dd" [selected]="formMantraRole() === 'Dd'">Dd</option>
                <option value="Ds" [selected]="formMantraRole() === 'Ds'">Ds</option>
                <option value="B" [selected]="formMantraRole() === 'B'">B</option>
                <option value="E" [selected]="formMantraRole() === 'E'">E</option>
                <option value="M" [selected]="formMantraRole() === 'M'">M</option>
                <option value="C" [selected]="formMantraRole() === 'C'">C</option>
                <option value="T" [selected]="formMantraRole() === 'T'">T</option>
                <option value="W" [selected]="formMantraRole() === 'W'">W</option>
                <option value="A" [selected]="formMantraRole() === 'A'">A</option>
                <option value="Pc" [selected]="formMantraRole() === 'Pc'">Pc</option>
              </select>
              <p class="mt-1 text-xs" style="color:var(--color-text-secondary)">
                Gerarchia profondità: Por ← Dc/B/Dd/Ds ← E/M ← C ← T/W ← A/Pc
              </p>
            </div>

            <!-- Mark as validated -->
            <div>
              <label class="flex items-center gap-2 text-sm cursor-pointer"
                     style="color:var(--color-text-secondary)">
                <input type="checkbox"
                       [checked]="formValidated()"
                       (change)="onValidatedChange($event)" />
                ✅ Dati verificati e corretti
              </label>
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
  readonly formMantraRole = signal<string>('');
  readonly formValidated = signal(false);
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

  /** Type-safe check for the optional `resolvedFromHistory` field. */
  readonly showFromHistory = (item: PlayerIdMapping): boolean =>
    (item as unknown as Record<string, unknown>)['resolvedFromHistory'] === true;

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
      unresolvedOnly: this.unresolvedOnly() || undefined,
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
  readonly onFormMantraRoleChange = (e: Event) => {
    this.formMantraRole.set((e.target as HTMLSelectElement).value);
  };
  readonly onValidatedChange = (e: Event) => {
    this.formValidated.set((e.target as HTMLInputElement).checked);
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
    this.formMantraRole.set('');
    this.formValidated.set(false);
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
    if (this.formMantraRole()) body.ruoloPrimario = this.formMantraRole();
    if (this.formValidated()) body.dataValidated = true;
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

  // ── Export ────────────────────────────────────────────────────────────────
  readonly exportIdMap = () => {
    this.idMappingService.exportMappings().subscribe({
      next: (data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `player_id_map_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      },
      error: (e) => {
        console.error('Export failed', e);
      },
    });
  };

  // ── Run Auto Mapping ──────────────────────────────────────────────────────
  readonly runAutoMapping = () => {
    if (!confirm('Avviare il mapping automatico? Potrebbero volerci alcuni secondi.')) return;
    this.idMappingService.runAutoMapping({ seasonStart: this.selectedSeason() ?? undefined }).subscribe({
      next: (res) => {
        alert(`Mapping completato: ${res.matched} matched, ${res.unmatched} unmatched (${res.matchRatePct}%)`);
        this.loadData();
        this.loadStats();
      },
      error: (e) => {
        alert('Errore durante il mapping: ' + (e.error?.detail ?? e.message));
      },
    });
  };

  // ── Verify on FotMob ──────────────────────────────────────────────────────
  readonly verifyMode = signal(false);
  readonly verifyUrl = signal('');
  readonly verifyItem = signal<PlayerIdMapping | null>(null);
  readonly verifyResult = signal<{ id: number; name: string } | null>(null);

  readonly verifyOnFotmob = (item: PlayerIdMapping) => {
    this.verifyItem.set(item);
    this.verifyMode.set(true);
    this.verifyUrl.set('');
    this.verifyResult.set(null);
    const query = encodeURIComponent(item.nameFantacalcio);
    window.open(`https://www.fotmob.com/search?q=${query}`, '_blank');
  };

  readonly parseFotmobUrl = () => {
    const url = this.verifyUrl();
    const match = url.match(/\/players\/(\d+)\/([a-z0-9-]+)/i);
    if (!match) {
      alert('URL non valida. Deve essere del tipo: https://www.fotmob.com/players/314605/alberto-moreno');
      return;
    }
    const id = parseInt(match[1], 10);
    const name = match[2]
      .split('-')
      .map((w: string) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
    this.verifyResult.set({ id, name });
  };

  readonly useVerifyResult = () => {
    const res = this.verifyResult();
    const item = this.verifyItem();
    if (!res || !item) return;
    this.openResolver(item);
    this.formFotmobId.set(res.id);
    this.formFotmobName.set(res.name);
    this.verifyMode.set(false);
  };
}
