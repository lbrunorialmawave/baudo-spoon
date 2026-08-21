import {
  Component,
  DestroyRef,
  afterNextRender,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { finalize } from 'rxjs/operators';
import { MyTeamService } from '../../core/services/my-team.service';
import { MANTRA_MODULE_LABELS } from '../../core/constants/shared-presets';
import {
  FormationAlternative,
  LineupOptimizeResponse,
  RosterImportResponse,
  RosterTeamCard,
  Ruleset,
  TradesDashboardResponse,
  TradeExecuteResponse,
} from '../../core/models/my-team.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { FieldLegendComponent, FieldLegendExample } from '../../shared/components/field-legend/field-legend.component';
import { PitchFieldComponent, toPitchPlayers } from './pitch-field/pitch-field.component';
import { TradeEvaluatorComponent } from './trade-evaluator/trade-evaluator.component';
import { canSwap, swapStarterWithBench } from './lineup-swap';
import {
  clearMyTeamSession,
  isContextMissingError,
  loadMyTeamSession,
  loadRosterFile,
  saveMyTeamSession,
  saveRosterFile,
  sessionDaysRemaining,
  type MyTeamSessionSnapshot,
} from './my-team-session';

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
    FieldLegendComponent,
    PitchFieldComponent,
    TradeEvaluatorComponent,
  ],
  template: `
    <div class="mx-auto max-w-6xl px-4 py-6 space-y-6">
      <header class="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 class="text-2xl font-semibold tracking-tight">La Mia Squadra</h1>
          <p class="text-sm opacity-70 mt-1">
            Import rose Fantagazzetta · ottimizza formazione · cruscotto scambi
          </p>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          @if (contextId()) {
            <div class="text-xs opacity-60 font-mono">
              context {{ contextId()!.slice(0, 8) }}…
            </div>
          }
          @if (hasPersistedSession()) {
            <button
              type="button"
              class="rounded-lg border px-3 py-1.5 text-xs font-medium hover:border-emerald-500"
              (click)="requestNewUpload()"
              title="Cancella la sessione salvata e carica un nuovo Excel"
            >
              Carica nuovo file
            </button>
          }
        </div>
      </header>

      @if (sessionHydrating()) {
        <div class="rounded-lg border border-dashed px-3 py-2 text-xs opacity-70">
          Ripristino sessione salvata…
        </div>
      }

      @if (sessionBanner(); as banner) {
        <div
          class="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-xs"
        >
          <span>
            Sessione salvata
            @if (banner.filename) {
              · <span class="font-medium">{{ banner.filename }}</span>
            }
            · valida ancora {{ banner.daysLeft }}
            {{ banner.daysLeft === 1 ? 'giorno' : 'giorni' }}
            @if (banner.restored) {
              · ripristinata da questo dispositivo
            }
          </span>
          <button
            type="button"
            class="underline opacity-80 hover:opacity-100"
            (click)="requestNewUpload()"
          >
            Sostituisci file
          </button>
        </div>
      }

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
            <button type="button" class="rounded-lg border px-4 py-2 text-sm" (click)="requestNewUpload()">
              Carica nuovo file
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

                  @if (topAlternatives().length) {
                    <section class="mt-4 space-y-2" aria-labelledby="alts-heading">
                      <h3 id="alts-heading" class="text-sm font-medium">
                        Moduli alternativi considerati
                        <span class="opacity-60 font-normal">
                          · prime {{ topAlternatives().length }} di
                          {{ lu.alternativesConsidered.length }}
                        </span>
                      </h3>
                      <app-field-legend
                        fieldId="legend-alts"
                        description="Lista delle formazioni valutate dall'ottimizzatore
                                    oltre a quella scelta. Il confronto dei punteggi
                                    evidenzia quanto il modulo titolare sia superiore
                                    alle alternative. I moduli 'non fattibili' sono
                                    quelli esclusi per mancanza di copertura slot
                                    rispetto alla rosa corrente."
                        [examples]="alternativesExamples"
                      />
                      <ul class="grid gap-2 sm:grid-cols-3">
                        @for (a of topAlternatives(); track a.formation) {
                          <li
                            class="rounded-lg border p-2 text-sm space-y-1"
                            [class.border-emerald-500]="a.formation === lu.chosenFormation"
                            [class.opacity-60]="!a.feasible"
                          >
                            <div class="flex justify-between items-baseline">
                              <span class="font-medium">{{ a.formation }}</span>
                              <span
                                class="text-xs px-1.5 py-0.5 rounded"
                                [class.bg-emerald-500]="a.feasible"
                                [class.text-white]="a.feasible"
                                [class.bg-amber-500]="!a.feasible"
                                [class.text-black]="!a.feasible"
                              >
                                {{ a.feasible ? 'fattibile' : 'non fattibile' }}
                              </span>
                            </div>
                            <div class="tabular-nums opacity-80">
                              score {{ a.scoreTotale | number: '1.2-2' }}
                            </div>
                            @if (a.reason) {
                              <div class="text-xs opacity-60">{{ a.reason }}</div>
                            }
                          </li>
                        }
                      </ul>
                    </section>
                  }
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
                <section class="space-y-2">
                  <h3 class="text-sm font-medium">Moduli da analizzare</h3>
                  <p class="text-xs opacity-60">
                    Seleziona i moduli MANTRA da usare per il calcolo di copertura
                    e per i candidati scambio. Default allineato al backend
                    (4-3-3, 3-5-2, 3-4-3).
                  </p>
                  <div
                    class="flex flex-wrap gap-2"
                    role="group"
                    aria-label="Selezione moduli MANTRA"
                    id="formation-prefs-group"
                  >
                    @for (m of mantraModuleLabels; track m) {
                      <label
                        class="cursor-pointer rounded-full border px-3 py-1 text-xs flex items-center gap-1"
                        [class.border-emerald-500]="isFormationSelected(m)"
                        [class.bg-emerald-500]="isFormationSelected(m)"
                        [class.text-white]="isFormationSelected(m)"
                      >
                        <input
                          type="checkbox"
                          class="hidden"
                          [checked]="isFormationSelected(m)"
                          (change)="toggleFormationPref(m)"
                        />
                        {{ m }}
                      </label>
                    }
                  </div>
                  <app-field-legend
                    fieldId="legend-formation-prefs"
                    description="Ogni modulo selezionato viene valutato per la copertura
                                della rosa. I target in entrata vengono filtrati in base
                                agli slot deficitari dei moduli scelti. La modifica
                                invalida la cache della dashboard e rilancia la request."
                    [examples]="formationPrefsExamples"
                  />
                </section>

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

                  <app-trade-evaluator
                    [contextId]="contextId()!"
                    [sheetName]="claimedSheet()"
                    [teamName]="claimedTeamName()"
                    [formationPrefs]="formationPrefs()"
                    [ruleset]="ruleset()"
                    [trades]="tr"
                    [counterpartyTeams]="opponents()"
                  />

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

  /** Client-side session expiry (ms epoch); null if nothing persisted. */
  readonly sessionExpiresAt = signal<number | null>(null);
  /** True after a successful restore from localStorage this page load. */
  readonly sessionRestored = signal(false);
  /** True while probing / re-importing a saved session. */
  readonly sessionHydrating = signal(false);

  readonly hasPersistedSession = computed(
    () => this.sessionExpiresAt() != null && this.importResult() != null,
  );

  readonly sessionBanner = computed(() => {
    const exp = this.sessionExpiresAt();
    if (exp == null || !this.importResult()) return null;
    return {
      filename: this.selectedFileName() || this.importResult()?.sourceFilename || null,
      daysLeft: sessionDaysRemaining(exp),
      restored: this.sessionRestored(),
    };
  });

  constructor() {
    // Browser-only: restore session after first render (SSR-safe).
    afterNextRender(() => {
      void this.restoreSession();
    });
  }

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

  /** Default allineato a `TradesDashboardRequest.formation_prefs` in `trades.py`. */
  static readonly DEFAULT_TRADES_FORMATION_PREFS: readonly string[] = [
    '4-3-3',
    '3-5-2',
    '3-4-3',
  ] as const;

  /** Moduli MANTRA selezionati per la dashboard scambi (modificabili da UI). */
  readonly formationPrefs = signal<string[]>([
    ...MyTeamComponent.DEFAULT_TRADES_FORMATION_PREFS,
  ]);
  /** Catalogo completo moduli MANTRA 2026/27 esposto al template. */
  readonly mantraModuleLabels = MANTRA_MODULE_LABELS;
  /** Esempi statici per la legenda del campo formationPrefs. */
  readonly formationPrefsExamples: readonly FieldLegendExample[] = [
    { label: '3 moduli', value: 'default bilanciato' },
    { label: '5+ moduli', value: 'analisi ampia, più candidati' },
    { label: '1 modulo', value: 'focus chirurgico su un assetto' },
  ];

  /** Esempi statici per la legenda delle alternative di formazione. */
  readonly alternativesExamples: readonly FieldLegendExample[] = [
    { label: '4-3-3', value: 'modulo scelto, score migliore' },
    { label: '3-5-2', value: 'alternativa con voto simile' },
    { label: '3-4-3', value: 'alternativa penalizzata da score più basso' },
  ];

  /**
   * Numero massimo di alternative di formazione mostrate nella UI.
   * Coerente con il default del backend (tutte le alternative) ma limitato
   * per leggibilità: vengono ordinate per score decrescente.
   */
  static readonly MAX_ALTERNATIVES_DISPLAYED = 3;

  readonly divisions = computed(() => this.importResult()?.divisions ?? []);
  readonly pitchPlayers = computed(() => toPitchPlayers(this.lineup()?.startingXi));

  /**
   * Top-N alternative di formazione ordinate per score decrescente.
   * Mostra prima i moduli fattibili (più utili per il confronto) e poi
   * quelli non fattibili come diagnostica.
   */
  readonly topAlternatives = computed<FormationAlternative[]>(() => {
    const alts = this.lineup()?.alternativesConsidered ?? [];
    if (!alts.length) return [];
    const sorted = [...alts].sort((a, b) => {
      if (a.feasible !== b.feasible) return a.feasible ? -1 : 1;
      return b.scoreTotale - a.scoreTotale;
    });
    return sorted.slice(0, MyTeamComponent.MAX_ALTERNATIVES_DISPLAYED);
  });

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
          void this.onImportSuccess(res, file, /*keepClaim*/ false);
        },
        error: (err) => {
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Import fallito',
          );
        },
      });
  }

  /**
   * Apply import result to UI state, persist session + Excel blob.
   * @param keepClaim when re-hydrating after silent re-import, preserve claimed team.
   */
  private async onImportSuccess(
    res: RosterImportResponse,
    file: File | null,
    keepClaim: boolean,
  ): Promise<void> {
    this.importResult.set(res);
    this.contextId.set(res.contextId);
    if (file) {
      this.selectedFile.set(file);
      this.selectedFileName.set(file.name);
      await saveRosterFile(file);
    }

    const claimed =
      keepClaim && this.claimedSheet() && this.claimedTeamName()
        ? { sheetName: this.claimedSheet(), teamName: this.claimedTeamName() }
        : null;

    if (!keepClaim) {
      this.claimedSheet.set('');
      this.claimedTeamName.set('');
      this.opponents.set([]);
      this.lineup.set(null);
      this.trades.set(null);
      this.giveSelection.set([]);
      this.receiveSelection.set([]);
      this.lastTransfer.set(null);
      this.counterpartyTeam = '';
      this.sessionRestored.set(false);
      this.step.set('select');
    }

    const snap = saveMyTeamSession({
      ruleset: this.ruleset(),
      importResult: res,
      claimed,
      sourceFilename:
        file?.name ?? res.sourceFilename ?? this.selectedFileName() ?? null,
    });
    this.sessionExpiresAt.set(snap.expiresAt);
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
          this.applyClaim(t.sheetName, t.teamName);
          this.persistSessionClaim(t.sheetName, t.teamName);
        },
        error: (err) => {
          if (isContextMissingError(err)) {
            void this.handleExpiredContext();
            return;
          }
          this.error.set(err?.error?.detail ?? err?.message ?? 'Claim fallito');
        },
      });
  }

  private applyClaim(sheetName: string, teamName: string): void {
    this.claimedSheet.set(sheetName);
    this.claimedTeamName.set(teamName);
    const opps = (this.importResult()?.teams ?? []).filter(
      (x) =>
        x.sheetName === sheetName &&
        x.teamName !== teamName &&
        !x.isEmpty,
    );
    this.opponents.set(opps);
    this.lineup.set(null);
    this.trades.set(null);
    this.giveSelection.set([]);
    this.receiveSelection.set([]);
    this.lastTransfer.set(null);
    this.counterpartyTeam = '';
    this.formationPrefs.set([...MyTeamComponent.DEFAULT_TRADES_FORMATION_PREFS]);
    this.tab.set('formation');
    this.step.set('workspace');
  }

  private persistSessionClaim(sheetName: string, teamName: string): void {
    const ir = this.importResult();
    if (!ir) return;
    const snap = saveMyTeamSession({
      ruleset: this.ruleset(),
      importResult: ir,
      claimed: { sheetName, teamName },
      sourceFilename:
        this.selectedFileName() || ir.sourceFilename || null,
    });
    this.sessionExpiresAt.set(snap.expiresAt);
  }

  /**
   * Restore last session from localStorage (+ Excel from IndexedDB if server context died).
   */
  private async restoreSession(): Promise<void> {
    const session = loadMyTeamSession();
    if (!session) return;

    this.sessionHydrating.set(true);
    this.error.set(null);
    try {
      this.ruleset.set(session.ruleset);
      this.importResult.set(session.importResult);
      this.contextId.set(session.importResult.contextId);
      this.selectedFileName.set(session.sourceFilename ?? '');
      this.sessionExpiresAt.set(session.expiresAt);
      this.sessionRestored.set(true);

      const alive = await this.probeContext(session.importResult.contextId);
      if (!alive) {
        const reimported = await this.reimportFromStoredFile(session);
        if (!reimported) {
          this.error.set(
            'Il context sul server è scaduto e non è disponibile una copia del file. Carica di nuovo l’Excel.',
          );
          this.step.set('upload');
          return;
        }
      }

      if (session.claimed) {
        await this.reclaimAfterRestore(
          session.claimed.sheetName,
          session.claimed.teamName,
        );
      } else {
        this.step.set('select');
      }
    } catch (err) {
      console.warn('[my-team] restoreSession failed', err);
      this.step.set('upload');
    } finally {
      this.sessionHydrating.set(false);
    }
  }

  private async probeContext(contextId: string): Promise<boolean> {
    try {
      await firstValueFrom(this.api.listTeams(contextId));
      return true;
    } catch {
      // Any failure → treat as dead context and attempt silent re-import.
      return false;
    }
  }

  private async reimportFromStoredFile(
    session: MyTeamSessionSnapshot,
  ): Promise<boolean> {
    const file = await loadRosterFile();
    if (!file) return false;
    try {
      this.loading.set(true);
      const res = await firstValueFrom(this.api.importRoster(file));
      if (session.claimed) {
        this.claimedSheet.set(session.claimed.sheetName);
        this.claimedTeamName.set(session.claimed.teamName);
      }
      await this.onImportSuccess(res, file, /*keepClaim*/ !!session.claimed);
      return true;
    } catch (err) {
      this.error.set(
        (err as { error?: { detail?: string }; message?: string })?.error
          ?.detail ??
          (err as { message?: string })?.message ??
          'Re-import automatico fallito',
      );
      return false;
    } finally {
      this.loading.set(false);
    }
  }

  private async reclaimAfterRestore(
    sheetName: string,
    teamName: string,
  ): Promise<void> {
    const cid = this.contextId();
    if (!cid) {
      this.step.set('select');
      return;
    }
    try {
      this.loading.set(true);
      await firstValueFrom(this.api.claimTeam(cid, sheetName, teamName));
      this.applyClaim(sheetName, teamName);
      this.persistSessionClaim(sheetName, teamName);
    } catch (err) {
      if (isContextMissingError(err)) {
        await this.handleExpiredContext();
        return;
      }
      // Claim failed for other reasons — fall back to team picker.
      this.error.set(
        (err as { error?: { detail?: string } })?.error?.detail ??
          'Impossibile ripristinare la squadra selezionata',
      );
      this.step.set('select');
    } finally {
      this.loading.set(false);
    }
  }

  /** Context died mid-session: try silent re-import, else force upload. */
  private async handleExpiredContext(): Promise<void> {
    const session = loadMyTeamSession();
    if (session) {
      const ok = await this.reimportFromStoredFile(session);
      if (ok && session.claimed) {
        await this.reclaimAfterRestore(
          session.claimed.sheetName,
          session.claimed.teamName,
        );
        return;
      }
    }
    this.error.set(
      'Context scaduto sul server. Carica di nuovo il file Excel.',
    );
    this.requestNewUpload(/*preserveError*/ true);
  }

  /**
   * Clear persisted session and return to the upload step.
   * @param preserveError when true, do not wipe the current error banner.
   */
  requestNewUpload(preserveError = false): void {
    clearMyTeamSession();
    this.sessionExpiresAt.set(null);
    this.sessionRestored.set(false);
    this.importResult.set(null);
    this.contextId.set(null);
    this.claimedSheet.set('');
    this.claimedTeamName.set('');
    this.opponents.set([]);
    this.lineup.set(null);
    this.trades.set(null);
    this.giveSelection.set([]);
    this.receiveSelection.set([]);
    this.lastTransfer.set(null);
    this.counterpartyTeam = '';
    this.selectedFile.set(null);
    this.selectedFileName.set('');
    this.divisionFilter.set(null);
    if (!preserveError) this.error.set(null);
    this.step.set('upload');
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
          if (isContextMissingError(err)) {
            void this.handleExpiredContext();
            return;
          }
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
    if (!this.contextId() || this.trades()) return;
    this.refreshTrades();
  }

  /** Fetch esplicito della dashboard scambi con i `formationPrefs` correnti. */
  refreshTrades(): void {
    const cid = this.contextId();
    if (!cid) return;
    this.loading.set(true);
    this.error.set(null);
    this.api
      .tradesDashboard({
        contextId: cid,
        sheetName: this.claimedSheet(),
        teamName: this.claimedTeamName(),
        formationPrefs: [...this.formationPrefs()],
      })
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        next: (res) => this.trades.set(res),
        error: (err) => {
          if (isContextMissingError(err)) {
            void this.handleExpiredContext();
            return;
          }
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Dashboard scambi fallita',
          );
        },
      });
  }

  isFormationSelected(moduleLabel: string): boolean {
    return this.formationPrefs().includes(moduleLabel);
  }

  /**
   * Toggle di un modulo nei `formationPrefs`.
   * Se la dashboard era già stata caricata, invalida la cache e rilancia
   * la request con il nuovo set di moduli.
   */
  toggleFormationPref(moduleLabel: string): void {
    const cur = this.formationPrefs();
    const next = cur.includes(moduleLabel)
      ? cur.filter((f) => f !== moduleLabel)
      : [...cur, moduleLabel];
    this.formationPrefs.set(next);
    if (this.trades() !== null && this.contextId()) {
      this.trades.set(null);
      this.refreshTrades();
    }
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
          if (isContextMissingError(err)) {
            void this.handleExpiredContext();
            return;
          }
          this.error.set(
            err?.error?.detail ?? err?.message ?? 'Esecuzione scambio fallita',
          );
        },
      });
  }

}
