import {
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';
import { MyTeamService } from '../../core/services/my-team.service';
import {
  LineupOptimizeResponse,
  RosterImportResponse,
  RosterTeamCard,
  Ruleset,
  TradesDashboardResponse,
  TradeExecuteResponse,
} from '../../core/models/my-team.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { PitchFieldComponent, toPitchPlayers } from './pitch-field/pitch-field.component';
import { canSwap, swapStarterWithBench } from './lineup-swap';

type Step = 'ruleset' | 'upload' | 'select' | 'workspace';
type Tab = 'formation' | 'trades';

@Component({
  selector: 'app-my-team',
  standalone: true,
  imports: [
    FormsModule,
    DecimalPipe,
    PercentPipe,
    SkeletonComponent,
    ErrorBoundaryComponent,
    PitchFieldComponent,
  ],
  template: `
    <div class="mx-auto max-w-6xl px-4 py-6 space-y-6">
      <header class="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 class="text-2xl font-semibold tracking-tight">La Mia Squadra</h1>
          <p class="text-sm opacity-70 mt-1">
            Import rose Fantagazzetta · ottimizza formazione · cruscotto scambi
            (runtime-only, senza persistenza rosa)
          </p>
        </div>
        @if (contextId()) {
          <div class="text-xs opacity-60 font-mono">
            context {{ contextId()!.slice(0, 8) }}…
          </div>
        }
      </header>

      <!-- Step indicator -->
      <nav class="flex flex-wrap gap-2 text-sm">
        @for (s of stepMeta; track s.id) {
          <button
            type="button"
            class="rounded-full px-3 py-1 border transition"
            [class.bg-emerald-600]="step() === s.id"
            [class.text-white]="step() === s.id"
            [class.opacity-50]="stepIndex(s.id) > stepIndex(step())"
            [disabled]="stepIndex(s.id) > stepIndex(step())"
            (click)="goStep(s.id)"
          >
            {{ s.label }}
          </button>
        }
      </nav>

      @if (error()) {
        <app-error-boundary title="Errore" [message]="error()!" />
      }

        <!-- Step 1: ruleset -->
        @if (step() === 'ruleset') {
          <section class="rounded-xl border p-6 space-y-4">
            <h2 class="text-lg font-medium">Modalità di gioco</h2>
            <div class="flex flex-wrap gap-3">
              <button
                type="button"
                class="rounded-lg border px-5 py-3 min-w-[140px] hover:border-emerald-500"
                [class.ring-2]="ruleset() === 'MANTRA'"
                [class.ring-emerald-500]="ruleset() === 'MANTRA'"
                (click)="ruleset.set('MANTRA')"
              >
                <div class="font-semibold">Mantra</div>
                <div class="text-xs opacity-70">12 ruoli · 11 moduli</div>
              </button>
              <button
                type="button"
                class="rounded-lg border px-5 py-3 min-w-[140px] opacity-60 cursor-not-allowed"
                title="Classic in arrivo"
                disabled
              >
                <div class="font-semibold">Classic</div>
                <div class="text-xs">Presto</div>
              </button>
            </div>
            <button
              type="button"
              class="rounded-lg bg-emerald-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-40"
              [disabled]="ruleset() !== 'MANTRA'"
              (click)="step.set('upload')"
            >
              Continua
            </button>
          </section>
        }

        <!-- Step 2: upload -->
        @if (step() === 'upload') {
          <section class="rounded-xl border p-6 space-y-4">
            <h2 class="text-lg font-medium">Carica export rose</h2>
            <p class="text-sm opacity-70">
              File Excel Fantagazzetta (blocchi multi-squadra, una o più divisioni).
            </p>
            <label
              class="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-10 cursor-pointer hover:border-emerald-500"
              (dragover)="$event.preventDefault()"
              (drop)="onDrop($event)"
            >
              <span class="text-sm">Trascina qui o clicca per scegliere</span>
              <span class="text-xs opacity-60">{{ selectedFileName() || '.xlsx' }}</span>
              <input
                type="file"
                accept=".xlsx,.xlsm,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                class="hidden"
                (change)="onFileInput($event)"
              />
            </label>
            <div class="flex gap-2">
              <button
                type="button"
                class="rounded-lg bg-emerald-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-40"
                [disabled]="!selectedFile() || loading()"
                (click)="doImport()"
              >
                {{ loading() ? 'Import…' : 'Importa' }}
              </button>
              <button type="button" class="rounded-lg border px-4 py-2 text-sm" (click)="step.set('ruleset')">
                Indietro
              </button>
            </div>
            @if (loading()) {
              <app-skeleton height="72px" />
            }
          </section>
        }

        <!-- Step 3: select team -->
        @if (step() === 'select') {
          <section class="space-y-4">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <h2 class="text-lg font-medium">Seleziona la tua squadra</h2>
              @if (importResult(); as ir) {
                <div class="text-xs opacity-70">
                  Match {{ ir.quality.matchRate | percent: '1.0-0' }}
                  · {{ ir.quality.auto }} auto · {{ ir.quality.provisional }} da verificare
                  · {{ ir.quality.unmatched }} unmatched
                </div>
              }
            </div>

            @if (divisions().length > 1) {
              <div class="flex flex-wrap gap-2">
                <button
                  type="button"
                  class="rounded-full px-3 py-1 text-sm border"
                  [class.bg-emerald-600]="!divisionFilter()"
                  [class.text-white]="!divisionFilter()"
                  (click)="divisionFilter.set(null)"
                >
                  Tutte
                </button>
                @for (d of divisions(); track d) {
                  <button
                    type="button"
                    class="rounded-full px-3 py-1 text-sm border"
                    [class.bg-emerald-600]="divisionFilter() === d"
                    [class.text-white]="divisionFilter() === d"
                    (click)="divisionFilter.set(d)"
                  >
                    {{ d }}
                  </button>
                }
              </div>
            }

            <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              @for (t of filteredTeams(); track t.sheetName + t.teamName) {
                <button
                  type="button"
                  class="text-left rounded-xl border p-4 hover:border-emerald-500 transition disabled:opacity-40"
                  [disabled]="t.isEmpty || loading()"
                  (click)="claim(t)"
                >
                  <div class="font-medium truncate">{{ t.teamName }}</div>
                  <div class="text-xs opacity-60 mt-1">{{ t.sheetName }}</div>
                  <div class="mt-3 flex justify-between text-sm">
                    <span>{{ t.playerCount }} giocatori</span>
                    <span>{{ t.totalSpent }} cr</span>
                  </div>
                  <div class="text-xs mt-1 opacity-70">
                    match {{ t.matchRate | percent: '1.0-0' }}
                    @if (t.isEmpty) {
                      <span class="text-amber-600"> · rosa vuota</span>
                    }
                  </div>
                </button>
              }
            </div>
            <button type="button" class="rounded-lg border px-4 py-2 text-sm" (click)="step.set('upload')">
              Altro file
            </button>
          </section>
        }

        <!-- Step 4: workspace tabs -->
        @if (step() === 'workspace') {
          <section class="space-y-4">
            <div class="flex flex-wrap items-center gap-3 justify-between">
              <div>
                <h2 class="text-lg font-medium">{{ claimedTeamName() }}</h2>
                <p class="text-xs opacity-60">{{ claimedSheet() }}</p>
              </div>
              <div class="flex gap-2 text-sm">
                <button
                  type="button"
                  class="rounded-lg px-3 py-1.5 border"
                  [class.bg-emerald-600]="tab() === 'formation'"
                  [class.text-white]="tab() === 'formation'"
                  (click)="tab.set('formation')"
                >
                  Formazione
                </button>
                <button
                  type="button"
                  class="rounded-lg px-3 py-1.5 border"
                  [class.bg-emerald-600]="tab() === 'trades'"
                  [class.text-white]="tab() === 'trades'"
                  (click)="openTrades()"
                >
                  Scambi
                </button>
              </div>
            </div>

            @if (tab() === 'formation') {
              <div class="rounded-xl border p-4 space-y-4">
                <div class="flex flex-wrap gap-3 items-end">
                  <label class="text-sm space-y-1">
                    <span class="opacity-70">Avversario (stessa divisione)</span>
                    <select
                      class="block rounded-lg border px-3 py-2 bg-transparent min-w-[200px]"
                      [(ngModel)]="opponentTeamName"
                    >
                      <option value="">— nessuno —</option>
                      @for (o of opponents(); track o.teamName) {
                        <option [value]="o.teamName">{{ o.teamName }}</option>
                      }
                    </select>
                  </label>
                  <button
                    type="button"
                    class="rounded-lg bg-emerald-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-40"
                    [disabled]="loading()"
                    (click)="doOptimize()"
                  >
                    {{ loading() ? 'Calcolo…' : 'Ottimizza formazione' }}
                  </button>
                </div>

                @if (lineup(); as lu) {
                  <div class="grid gap-4 lg:grid-cols-3">
                    <div class="lg:col-span-1">
                      <app-pitch-field
                        [formation]="lu.chosenFormation || null"
                        [players]="pitchPlayers()"
                        [score]="lu.scoreTotale ?? null"
                      />
                    </div>
                    <div>
                      <div class="text-sm font-medium mb-2">
                        {{ lu.chosenFormation || 'Nessun modulo fattibile' }}
                        @if (lu.scoreTotale != null) {
                          <span class="opacity-70 font-normal">
                            · score {{ lu.scoreTotale | number: '1.1-1' }}
                          </span>
                        }
                      </div>
                      <ul class="space-y-1 text-sm">
                        @for (s of lu.startingXi; track s.playerId + s.slotLabel) {
                          <li
                            class="flex justify-between gap-2 border-b border-white/5 py-1 cursor-pointer rounded px-1"
                            [class.ring-1]="selectedStarterId() === s.playerId"
                            [class.ring-emerald-400]="selectedStarterId() === s.playerId"
                            (click)="selectStarter(s.playerId)"
                            title="Seleziona per sostituzione"
                          >
                            <span>
                              <span class="opacity-50 text-xs mr-1">{{ s.slotLabel }}</span>
                              {{ s.playerName }}
                            </span>
                            <span class="tabular-nums opacity-80">
                              {{ s.expectedScore | number: '1.2-2' }}
                            </span>
                          </li>
                        }
                      </ul>
                    </div>
                    <div>
                      <div class="text-sm font-medium mb-2">Panchina</div>
                      <ul class="space-y-1 text-sm max-h-64 overflow-auto">
                        <p class="text-xs opacity-60 mb-1">
                          @if (selectedStarterId()) {
                            Clicca un panchinaro compatibile per sostituire
                          } @else {
                            Seleziona un titolare, poi un panchinaro
                          }
                        </p>
                        @for (b of lu.bench; track b.playerId) {
                          <li
                            class="flex justify-between gap-2 py-0.5 rounded px-1"
                            [class.opacity-40]="selectedStarterId() && !benchCompatible(b)"
                            [class.cursor-pointer]="!selectedStarterId() || benchCompatible(b)"
                            [class.hover:bg-white/5]="!selectedStarterId() || benchCompatible(b)"
                            (click)="trySwapWithBench(b.playerId)"
                          >
                            <span>{{ b.playerName }}
                              <span class="text-xs opacity-50">{{ b.slotRoles?.join('/') }}</span>
                            </span>
                            <span class="tabular-nums">{{ b.expectedScore | number: '1.2-2' }}</span>
                          </li>
                        }
                      </ul>
                      @if (lu.notes?.length) {
                        <ul class="mt-3 text-xs opacity-60 list-disc pl-4">
                          @for (n of lu.notes; track n) {
                            <li>{{ n }}</li>
                          }
                        </ul>
                      }
                    </div>
                  </div>
                } @else if (loading()) {
                  <app-skeleton height="160px" />
                } @else {
                  <p class="text-sm opacity-60">
                    Scegli un avversario (opzionale) e premi Ottimizza.
                  </p>
                }
              </div>
            }

            @if (tab() === 'trades') {
              <div class="rounded-xl border p-4 space-y-4">
                @if (loading() && !trades()) {
                  <app-skeleton height="120px" />
                }
                @if (trades(); as tr) {
                  <div>
                    <h3 class="text-sm font-medium mb-2">Copertura moduli</h3>
                    <div class="flex flex-wrap gap-2 text-xs">
                      @for (f of tr.formationPrefs; track f) {
                        <span
                          class="rounded-full px-2 py-1 border"
                          [class.border-emerald-500]="tr.coverageByFormation[f]"
                          [class.border-amber-500]="!tr.coverageByFormation[f]"
                        >
                          {{ f }}
                          {{ tr.coverageByFormation[f] ? '✓' : 'deficit' }}
                        </span>
                      }
                    </div>
                  </div>

                  <div class="grid gap-4 lg:grid-cols-2">
                    <div>
                      <h3 class="text-sm font-medium mb-2">Candidati in uscita</h3>
                      <ul class="space-y-2 text-sm">
                        @for (c of tr.tradeOutCandidates; track c.player.playerId) {
                          <li class="rounded-lg border p-2">
                            <label class="flex gap-2 items-start cursor-pointer">
                              <input
                                type="checkbox"
                                class="mt-1"
                                [checked]="isGiveSelected(c.player.playerId)"
                                (change)="toggleGive(c.player)"
                              />
                              <span>
                                <div class="font-medium">{{ c.player.name }}</div>
                                <div class="text-xs opacity-70">
                                  retention {{ c.retentionScore | number: '1.0-1' }}
                                  · {{ c.surplusRoles.join(', ') }}
                                  · {{ c.player.cost }} cr
                                </div>
                                <div class="text-xs opacity-60 mt-1">{{ c.rationale }}</div>
                              </span>
                            </label>
                          </li>
                        } @empty {
                          <li class="text-xs opacity-60">Nessun surplus sotto soglia esclusione</li>
                        }
                      </ul>
                      @if (tr.excludedTopPerformers.length) {
                        <h3 class="text-sm font-medium mt-4 mb-2">Esclusi (top performer)</h3>
                        <ul class="space-y-1 text-xs opacity-80">
                          @for (e of tr.excludedTopPerformers; track e.player.playerId) {
                            <li>{{ e.player.name }} — {{ e.reason }}</li>
                          }
                        </ul>
                      }
                    </div>
                    <div>
                      <h3 class="text-sm font-medium mb-2">Obiettivi in entrata</h3>
                      <ul class="space-y-2 text-sm">
                        @for (t of tr.tradeInTargets; track t.playerId) {
                          <li class="rounded-lg border p-2">
                            <label class="flex gap-2 items-start cursor-pointer">
                              <input
                                type="checkbox"
                                class="mt-1"
                                [checked]="isReceiveSelected(t.playerId)"
                                (change)="toggleReceive(t)"
                              />
                              <span>
                                <div class="font-medium">{{ t.name }}</div>
                                <div class="text-xs opacity-70">
                                  copre {{ t.coversSlots.join(', ') }}
                                  · FP {{ t.fpCorr | number: '1.0-0' }}
                                  · ~{{ t.estimatedCost }} cr
                                </div>
                              </span>
                            </label>
                          </li>
                        } @empty {
                          <li class="text-xs opacity-60">Nessun target (o nessun deficit)</li>
                        }
                      </ul>
                    </div>
                  </div>
                  <div class="rounded-lg border border-dashed p-4 space-y-3">
                    <h3 class="text-sm font-medium">Esegui scambio</h3>
                    <label class="block text-sm space-y-1">
                      <span class="opacity-70">Controparte (stessa divisione)</span>
                      <select
                        class="block w-full max-w-md rounded-lg border px-3 py-2 bg-transparent"
                        [(ngModel)]="counterpartyTeam"
                      >
                        <option value="">— seleziona —</option>
                        @for (o of opponents(); track o.teamName) {
                          <option [value]="o.teamName">{{ o.teamName }}</option>
                        }
                      </select>
                    </label>
                    <label class="flex items-center gap-2 text-sm">
                      <input type="checkbox" [(ngModel)]="penaltyEnabled" />
                      Penalità crediti ({{ decayPercent }}% / floor {{ floorPercent }}%)
                    </label>
                    <div class="text-xs opacity-70">
                      Dai: {{ giveSelection().length }} · Ricevi: {{ receiveSelection().length }}
                    </div>
                    <button
                      type="button"
                      class="rounded-lg bg-emerald-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-40"
                      [disabled]="loading() || !canExecuteTrade()"
                      (click)="doExecuteTrade()"
                    >
                      {{ loading() ? 'Invio…' : 'Registra scambio' }}
                    </button>
                    @if (lastTransfer(); as lt) {
                      <div class="text-xs rounded-lg border p-3 space-y-1">
                        <div class="font-medium text-emerald-500">Scambio {{ lt.transferId.slice(0, 8) }}…</div>
                        @for (leg of lt.legs; track leg.playerId + leg.direction) {
                          <div>
                            {{ leg.direction }} {{ leg.playerId }}:
                            {{ leg.valueBefore }} → {{ leg.valueAfter }}
                            @if (leg.penaltyApplied) { <span>(penalità)</span> }
                          </div>
                        }
                      </div>
                    }
                  </div>

                  @if (tr.notes?.length) {
                    <ul class="text-xs opacity-60 list-disc pl-4">
                      @for (n of tr.notes; track n) {
                        <li>{{ n }}</li>
                      }
                    </ul>
                  }
                }
              </div>
            }
          </section>
        }
    </div>
  `,
})
export class MyTeamComponent {
  private readonly api = inject(MyTeamService);
  private readonly destroyRef = inject(DestroyRef);

  readonly stepMeta = [
    { id: 'ruleset' as Step, label: '1. Modalità' },
    { id: 'upload' as Step, label: '2. Upload' },
    { id: 'select' as Step, label: '3. Squadra' },
    { id: 'workspace' as Step, label: '4. Workspace' },
  ];

  readonly step = signal<Step>('ruleset');
  readonly tab = signal<Tab>('formation');
  readonly ruleset = signal<Ruleset>('MANTRA');
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  readonly selectedFile = signal<File | null>(null);
  readonly selectedFileName = signal<string>('');
  readonly importResult = signal<RosterImportResponse | null>(null);
  readonly contextId = signal<string | null>(null);
  readonly divisionFilter = signal<string | null>(null);

  readonly claimedSheet = signal<string>('');
  readonly claimedTeamName = signal<string>('');
  readonly opponents = signal<RosterTeamCard[]>([]);

  opponentTeamName = '';

  readonly lineup = signal<LineupOptimizeResponse | null>(null);
  readonly trades = signal<TradesDashboardResponse | null>(null);
  readonly giveSelection = signal<Array<{ playerId: string; cost: number }>>([]);
  readonly receiveSelection = signal<Array<{ playerId: string; cost: number }>>([]);
  readonly lastTransfer = signal<TradeExecuteResponse | null>(null);
  readonly selectedStarterId = signal<string | null>(null);
  counterpartyTeam = '';
  penaltyEnabled = false;
  decayPercent = 25;
  floorPercent = 25;

  readonly divisions = computed(() => this.importResult()?.divisions ?? []);
  readonly pitchPlayers = computed(() => toPitchPlayers(this.lineup()?.startingXi));

  readonly filteredTeams = computed(() => {
    const teams = this.importResult()?.teams ?? [];
    const f = this.divisionFilter();
    return f ? teams.filter((t) => t.sheetName === f) : teams;
  });

  stepIndex(s: Step): number {
    return this.stepMeta.findIndex((x) => x.id === s);
  }

  goStep(s: Step): void {
    if (this.stepIndex(s) <= this.stepIndex(this.step())) {
      this.step.set(s);
    }
  }

  onFileInput(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.selectedFile.set(file);
    this.selectedFileName.set(file?.name ?? '');
  }

  onDrop(ev: DragEvent): void {
    ev.preventDefault();
    const file = ev.dataTransfer?.files?.[0] ?? null;
    if (file) {
      this.selectedFile.set(file);
      this.selectedFileName.set(file.name);
    }
  }

  doImport(): void {
    const file = this.selectedFile();
    if (!file) return;
    this.loading.set(true);
    this.error.set(null);
    this.api
      .importRoster(file)
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        next: (res) => {
          this.importResult.set(res);
          this.contextId.set(res.contextId);
          this.step.set('select');
        },
        error: (err) => {
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Import fallito',
          );
        },
      });
  }

  claim(t: RosterTeamCard): void {
    const cid = this.contextId();
    if (!cid) return;
    this.loading.set(true);
    this.error.set(null);
    this.api
      .claimTeam(cid, t.sheetName, t.teamName)
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        next: () => {
          this.claimedSheet.set(t.sheetName);
          this.claimedTeamName.set(t.teamName);
          const opps = (this.importResult()?.teams ?? []).filter(
            (x) =>
              x.sheetName === t.sheetName &&
              x.teamName !== t.teamName &&
              !x.isEmpty,
          );
          this.opponents.set(opps);
          this.lineup.set(null);
          this.trades.set(null);
          this.giveSelection.set([]);
          this.receiveSelection.set([]);
          this.lastTransfer.set(null);
          this.counterpartyTeam = '';
          this.tab.set('formation');
          this.step.set('workspace');
        },
        error: (err) => {
          this.error.set(err?.error?.detail ?? err?.message ?? 'Claim fallito');
        },
      });
  }

  doOptimize(): void {
    const cid = this.contextId();
    if (!cid) return;
    this.loading.set(true);
    this.error.set(null);
    this.api
      .optimizeLineup({
        contextId: cid,
        sheetName: this.claimedSheet(),
        teamName: this.claimedTeamName(),
        ruleset: this.ruleset(),
        opponentTeamName: this.opponentTeamName || null,
        opponentSheetName: this.opponentTeamName ? this.claimedSheet() : null,
      })
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        next: (res) => this.lineup.set(res),
        error: (err) => {
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Ottimizzazione fallita',
          );
        },
      });
  }


  selectStarter(playerId: string): void {
    this.selectedStarterId.set(
      this.selectedStarterId() === playerId ? null : playerId,
    );
  }

  benchCompatible(b: { playerId: string; slotLabel: string; slotRoles: string[] }): boolean {
    const lu = this.lineup();
    const sid = this.selectedStarterId();
    if (!lu || !sid) return true;
    const starter = lu.startingXi.find((s) => s.playerId === sid);
    if (!starter) return false;
    return canSwap(starter, b as any);
  }

  trySwapWithBench(benchPlayerId: string): void {
    const lu = this.lineup();
    const sid = this.selectedStarterId();
    if (!lu || !sid) return;
    const next = swapStarterWithBench(lu, sid, benchPlayerId);
    if (!next) {
      this.error.set('Ruoli non compatibili per questa sostituzione');
      return;
    }
    this.error.set(null);
    this.lineup.set(next);
    this.selectedStarterId.set(null);
  }

  openTrades(): void {
    this.tab.set('trades');
    const cid = this.contextId();
    if (!cid || this.trades()) return;
    this.loading.set(true);
    this.error.set(null);
    this.api
      .tradesDashboard({
        contextId: cid,
        sheetName: this.claimedSheet(),
        teamName: this.claimedTeamName(),
      })
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        next: (res) => this.trades.set(res),
        error: (err) => {
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Dashboard scambi fallita',
          );
        },
      });
  }
  isGiveSelected(id: string): boolean {
    return this.giveSelection().some((x) => x.playerId === id);
  }

  isReceiveSelected(id: string): boolean {
    return this.receiveSelection().some((x) => x.playerId === id);
  }

  toggleGive(player: { playerId: string; cost: number }): void {
    const cur = this.giveSelection();
    if (cur.some((x) => x.playerId === player.playerId)) {
      this.giveSelection.set(cur.filter((x) => x.playerId !== player.playerId));
    } else {
      this.giveSelection.set([...cur, { playerId: player.playerId, cost: player.cost }]);
    }
  }

  toggleReceive(t: { playerId: string; estimatedCost: number }): void {
    const cur = this.receiveSelection();
    if (cur.some((x) => x.playerId === t.playerId)) {
      this.receiveSelection.set(cur.filter((x) => x.playerId !== t.playerId));
    } else {
      this.receiveSelection.set([
        ...cur,
        { playerId: t.playerId, cost: t.estimatedCost },
      ]);
    }
  }

  canExecuteTrade(): boolean {
    return (
      !!this.counterpartyTeam &&
      (this.giveSelection().length > 0 || this.receiveSelection().length > 0)
    );
  }

  doExecuteTrade(): void {
    const cid = this.contextId();
    if (!cid || !this.canExecuteTrade()) return;
    this.loading.set(true);
    this.error.set(null);
    this.api
      .executeTrade({
        contextId: cid,
        sheetName: this.claimedSheet(),
        fromTeamName: this.claimedTeamName(),
        toTeamName: this.counterpartyTeam,
        give: this.giveSelection().map((g) => ({
          playerId: g.playerId,
          originalPurchasePrice: g.cost,
          currentValue: g.cost,
        })),
        receive: this.receiveSelection().map((r) => ({
          playerId: r.playerId,
          originalPurchasePrice: r.cost,
          currentValue: r.cost,
        })),
        creditPenaltyEnabled: this.penaltyEnabled,
        decayStepPercent: this.decayPercent,
        floorPercent: this.floorPercent,
      })
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        next: (res) => {
          this.lastTransfer.set(res);
          this.giveSelection.set([]);
          this.receiveSelection.set([]);
        },
        error: (err) => {
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Esecuzione scambio fallita',
          );
        },
      });
  }

}
