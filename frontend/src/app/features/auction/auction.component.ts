import { Component, computed, inject, signal, DestroyRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { forkJoin, Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { AuctionService } from '../../core/services/auction.service';
import { QuotationService } from '../../core/services/quotation.service';
import { SEASON_FALLBACK_LIST } from '../../core/constants/season-fallback.constant';
import {
  AUCTION_ROLES,
  MANTRA_DEFAULT_QUOTAS,
  MANTRA_ROLES,
  AssignmentRecord,
  AuctionConfig,
  AuctionParticipantSetup,
  AuctionParticipantState,
  AuctionPlayerSummary,
  AuctionRole,
  AuctionRuleset,
  AuctionSummary,
  AuctionTier,
  MantraModuleCoverage,
  ProjectionResponse,
  AlternativesResponse,
  ValuationMode,
  VarRankingItem,
} from '../../core/models/auction.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { AuctionPlayerDrawerComponent, AuctionDrawerPlayer } from './auction-player-drawer/auction-player-drawer.component';
import { AuctionSimulationComponent } from './auction-simulation/auction-simulation.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import {
  FieldLegendComponent,
  FieldLegendExample,
} from '../../shared/components/field-legend/field-legend.component';
import { OPTIMIZER_LEGENDS } from '../optimizer/optimizer.component';
import {
  AUCTION_PRESET_NONE,
  AUCTION_PRESETS,
  AuctionPreset,
  findAuctionPreset,
} from '../../core/constants/auction-presets';


const ROLE_COLOR: Record<string, string> = {
  P: 'var(--color-role-gk)',
  D: 'var(--color-role-def)',
  C: 'var(--color-role-mid)',
  A: 'var(--color-role-fwd)',
  // MANTRA 12-role palette (grouped by classic line)
  Por: 'var(--color-role-gk)',
  Dc: 'var(--color-role-def)',
  B: 'var(--color-role-def)',
  Dd: 'var(--color-role-def)',
  Ds: 'var(--color-role-def)',
  E: 'var(--color-role-mid)',
  M: 'var(--color-role-mid)',
  T: 'var(--color-role-fwd)',
  W: 'var(--color-role-fwd)',
  Pc: 'var(--color-role-fwd)',
};

const TIER_COLOR: Record<AuctionTier, string> = {
  LOW: 'var(--color-text-secondary)',
  MID: 'var(--color-accent)',
  TOP: '#F59E0B',
};

/** Colonne ordinabili della tabella "Ranking VAR/ESV (surplus vs expected_price)". */
type VarSortKey =
  | 'name'
  | 'role'
  | 'esv'
  | 'expectedPrice'
  | 'seasonValue'
  | 'startProbability'
  | 'buySignal';
/** Direzione di ordinamento: `null` = ordine naturale (non ordinato). */
type SortDir = 'asc' | 'desc' | null;

/**
 * Mappa delle legende visualizzate sotto ogni campo del pannello di
 * configurazione e della vista live. Ogni entry contiene una descrizione
 * estesa del significato e dell'uso del campo, più una lista di esempi
 * concreti per facilitare la comprensione a tutti gli utenti.
 *
 * Le chiavi corrispondono al `fieldId` utilizzato per `aria-describedby`
 * (es. `'legend-seasonStart'` → chiave `'seasonStart'`).
 */
const SETUP_LEGENDS: Readonly<
  Record<string, { description: string; examples: readonly FieldLegendExample[] }>
> = {
  seasonStart: {
    description:
      'Chiave di lookup DB: carica listini (qt_a), id-map e predizioni ML per costruire il player pool della sessione. Non è un parametro dell\'EWMA; senza stagione corretta il pool è vuoto o obsoleto.',
    examples: [
      { label: '2025', value: 'stagione 2025/26' },
      { label: '2024', value: 'stagione 2024/25' },
    ],
  },
  numParticipants: {
    description:
      'Numero di squadre in asta. Deve coincidere con len(participants) all\'init. Entra in AuctionConfig e, se useInflationBaseline è attivo, alimenta la curva di inflazione del baseline_cost (partecipanti oltre baselineParticipants).',
    examples: [
      { label: '8', value: 'default tipico' },
      { label: '10–12', value: 'più competizione → baseline_cost più alti se inflazione ON' },
    ],
  },
  defaultBudget: {
    description:
      'Scorciatoia UI: propaga questo valore a participants[].budgetInitial (crediti residui iniziali di ogni manager). Non è referenceBudget né il fattore di scala del listino.',
    examples: [
      { label: '300 cr.', value: 'lega listino ufficiale' },
      { label: '500 cr.', value: 'lega moderna tipica' },
    ],
  },
  roleQuotas: {
    description:
      'Quote hard di rosa CLASSIC per partecipante: quanti P/D/C/A può avere al massimo. Usate dal record assignment (rifiuta se il ruolo è pieno) e dal ranking VAR in modalità roster_depth. Totale tipico 3+8+8+6 = 25.',
    examples: [
      { label: 'P3 D8 C8 A6', value: 'default Fantacalcio classico (25 slot)' },
    ],
  },
  useInflationBaseline: {
    description:
      'Se ON, il baseline_cost usato dall\'EWMA (prezzo atteso pre-assegnazione) incorpora estimate_effective_cost (inflazione statica da percentile ruolo + partecipanti + eventuale Elo). Se OFF, il baseline resta ancorato al listino scalato, senza quella curva.',
    examples: [
      { label: 'ON', value: 'expected_price riflette competizione e rarità' },
      { label: 'OFF', value: 'expected_price più vicino al listino scalato' },
    ],
  },
  referenceBudget: {
    description:
      'Budget su cui è tarato il file quotazioni (storicamente 300). Il listino viene riproporzionato con il fattore budgetInitial / referenceBudget prima del drift. Esempio: listino 20 con ref=300 e budget=500 → baseline di partenza 20 × 500/300.',
    examples: [
      { label: '300', value: 'default listino ufficiale' },
      { label: '500', value: 'solo se il listino è già calibrato a 500' },
    ],
  },
  budgetInitial: {
    description:
      'Budget per squadra della sessione (config.budgetInitial): scala il listino insieme a referenceBudget e allinea le aspettative di prezzo al potere d\'acquisto reale. I crediti residui per manager restano in participants[].budgetInitial (spesso uguale).',
    examples: [
      { label: '500', value: 'lega a 500 cr.' },
      { label: '300', value: 'lega a listino ufficiale' },
    ],
  },
  valuationMode: {
    description:
      'Metrica usata dal ranking VAR/ESV in sessione. PER_MATCH_RATING ordina su projected_score (fantavoto/partita). SEASON_VALUE ordina su season_value (rating × presenze attese). Non cambia la formula EWMA del prezzo.',
    examples: [
      { label: 'PER_MATCH_RATING', value: 'default: rendimento a partita' },
      { label: 'SEASON_VALUE', value: 'valore totale di stagione proiettato' },
    ],
  },
  replacementMethod: {
    description:
      'Come si calcola il replacement level nel ranking VAR/ESV. "percentile" = basso percentile per ruolo nel pool; "roster_depth" = soglia legata a numParticipants × roleQuotas.',
    examples: [
      { label: 'percentile', value: 'default' },
      { label: 'roster_depth', value: 'legato alle quote di rosa di lega' },
    ],
  },
  minStartProbability: {
    description:
      'Filtro sul ranking VAR: i giocatori con start_probability sotto soglia restano nel pool d\'asta ma non compaiono nella classifica VAR/ESV. null = nessun filtro. Non blocca l\'assegnazione manuale.',
    examples: [
      { label: 'vuoto', value: 'nessun filtro (default)' },
      { label: '0.65', value: 'nasconde riserve chiare dal ranking affari' },
    ],
  },
  alpha: {
    description:
      'Peso EWMA sull\'ultimo ratio prezzo_pagato / expected_price: index ← (1−α)·index + α·ratio. α alto → l\'indice di ruolo×tier reagisce forte a ogni assegnazione; α basso → mercato più "lento". Bound: (0, 1]. Default codice 0.3.',
    examples: [
      { label: '0.20', value: 'mercato stabile / late sniper' },
      { label: '0.30', value: 'default engine' },
      { label: '0.50', value: 'reattivo / asta calda' },
    ],
  },
  spilloverAdj: {
    description:
      'Frazione dello shock di prezzo propagata ai tier adiacenti dello stesso ruolo (LOW↔MID↔TOP) dopo un update EWMA. 0 = isolato per tier. Default codice 0.25.',
    examples: [
      { label: '0.15', value: 'spillover contenuto' },
      { label: '0.25', value: 'default engine' },
      { label: '0.35', value: 'mercato molto liquido tra tier' },
    ],
  },
  spilloverCross: {
    description:
      'Hook di spillover verso altri ruoli dopo un update. Default codice 0 (disattivato): lascialo a 0 salvo leghe con contagio inter-ruolo esplicito. Non è legato alle conversioni MANTRA.',
    examples: [
      { label: '0.0', value: 'default: nessun contagio cross-ruolo' },
      { label: '0.05–0.10', value: 'contagio leggero (avanzato)' },
    ],
  },
  lowCostPercentile: {
    description:
      'Soglia sul expected_price (percentile nel ruolo) per il suggerimento low-cost in /alternatives. Sotto soglia un giocatore è candidato "economico" rispetto al target. Default codice 0.4.',
    examples: [
      { label: '0.30', value: 'alternative più aggressive sul prezzo' },
      { label: '0.40', value: 'default engine' },
      { label: '0.50', value: 'fascia media-bassa più ampia' },
    ],
  },
  minIndex: {
    description:
      'Floor dell\'indice EWMA per ogni cella ruolo×tier dopo clamp. Impedisce che index scenda sotto questo valore quando si pagano prezzi bassi rispetto all\'expected. Default codice 0.5.',
    examples: [
      { label: '0.50', value: 'default engine' },
      { label: '0.60', value: 'floor più alto (meno deflazione)' },
    ],
  },
  maxIndex: {
    description:
      'Cap dell\'indice EWMA per ruolo×tier. Blocca runaway quando si pagano multipli alti dell\'expected_price. expected_price ≈ baseline_cost × index. Default codice 1.8.',
    examples: [
      { label: '1.8', value: 'default engine' },
      { label: '2.0–2.2', value: 'asta molto calda' },
    ],
  },
  tierLow: {
    description:
      'Soglia bassa sui percentile di ruolo del giocatore (non sull\'indice EWMA): percentile < tierLow → tier LOW. Insieme a tierTop definisce MID. Classificazione usata per scegliere quale cella index[role][tier] aggiornare. Default codice 0.4.',
    examples: [
      { label: '0.40', value: 'default engine' },
      { label: '0.30', value: 'più giocatori finiscono in MID/TOP' },
    ],
  },
  tierTop: {
    description:
      'Soglia alta sul percentile di ruolo: percentile ≥ tierTop → tier TOP; tra tierLow e tierTop → MID. Deve essere > tierLow. Default codice 0.8.',
    examples: [
      { label: '0.80', value: 'default engine' },
      { label: '0.70', value: 'TOP più ampio' },
    ],
  },
};

const LIVE_LEGENDS: Readonly<
  Record<string, { description: string; examples: readonly FieldLegendExample[] }>
> = {
  lookupQuery: {
    description:
      'Ricerca substring sul pool ancora disponibile (GET /pool?q=…). Debounce 300 ms. Serve a risolvere nome → playerId per projection e alternatives, non modifica indici o budget.',
    examples: [
      { label: 'Lautaro', value: 'match sul nome' },
      { label: 'Invio', value: 'lookup su projection + alternatives' },
    ],
  },
  recordPlayer: {
    description:
      'playerId dell\'assegnazione da registrare (POST /record). Di solito precompilato dal lookup; deve essere un giocatore ancora nel pool.',
    examples: [
      { label: 'da Lookup', value: 'selezione dal dropdown' },
      { label: 'fm-…', value: 'ID manuale se noto' },
    ],
  },
  recordWinner: {
    description:
      'participantId del manager che si aggiudica il giocatore. Il backend verifica budget residuo, quote ruolo e disponibilità slot prima di accettare.',
    examples: [
      { label: 'Team N', value: 'vincitore del turno' },
    ],
  },
  recordPrice: {
    description:
      'Prezzo finale pagato (cr., intero ≥ 1). Scala il budget residuo del vincitore e aggiorna l\'EWMA: ratio = finalPrice / expectedPrice → index[role][tier] con α e spillover. È l\'unico input che muove gli indici di mercato.',
    examples: [
      { label: '1', value: 'minimo / svincolo' },
      { label: '≈ expected', value: 'indice resta stabile' },
      { label: '>> expected', value: 'indice sale (mercato caldo su quel tier)' },
    ],
  },
  varRanking: {
    description:
      'GET /var-ranking: classifica i disponibili per ESV (Expected Surplus Value = affare vs prezzo atteso), non "Expected Season Value". Colonne tipiche: var_score, expected_price, esv, buySignal. minStartProbability filtra solo questa vista, non il pool assegnabile.',
    examples: [
      { label: 'ESV > 0 + COMPRA', value: 'prezzo atteso basso vs contributo' },
      { label: 'ESV ≤ 0', value: 'caro rispetto al ranking corrente' },
    ],
  },
};

function makeParticipants(
  n: number,
  budget: number,
  existing: AuctionParticipantSetup[] = [],
): AuctionParticipantSetup[] {
  return Array.from(
    { length: n },
    (_, i) =>
      existing[i] ?? {
        participantId: `team_${i + 1}`,
        displayName: `Team ${i + 1}`,
        budgetInitial: budget,
      },
  );
}

@Component({
  selector: 'app-auction',
  standalone: true,
  imports: [
    FormsModule,
    DecimalPipe,
    SkeletonComponent,
    ErrorBoundaryComponent,
    FieldLegendComponent,
    AuctionPlayerDrawerComponent,
    AuctionSimulationComponent,
  ],
  template: `
    @if (sessionId()) {
      <!-- ═══════════════════════ LIVE VIEW ═══════════════════════ -->
      <div class="auction-page">
        <header class="auc-topbar">
          <div class="auc-topbar__brand">
            <h1 class="auc-topbar__title">Tracker Asta</h1>
            <p class="auc-topbar__subtitle">
              Sessione attiva · <code class="session-id">{{ sessionId()!.slice(0, 12) }}…</code>
              · cerca un giocatore e registra l’acquisto in un tap
              @if (nExcludedNoProjection() > 0) {
                · <span class="auc-exclusion-hint">{{ nExcludedNoProjection() }} senza proiezione</span>
              }
            </p>
          </div>
          <div class="auc-topbar__actions">
            <button
              type="button"
              class="secondary-btn"
              (click)="saveToFile()"
              title="Esporta sessione (assegnazioni, budget, indici EWMA) in JSON."
            >
              Salva
            </button>
            <button
              type="button"
              class="danger-btn"
              (click)="endSession()"
              title="Termina e cancella la sessione dal backend."
            >
              Termina
            </button>
          </div>
        </header>

        <!-- Price index strip -->
        @if (summary(); as s) {
          <div
            class="price-strip"
            title="Indice EWMA per ruolo × tier. Valori > 1 = mercato più caro del listino."
          >
            <span class="price-strip__title">Quanto costa il mercato</span>
            @for (role of displayRoles(); track role) {
              <div class="price-role-group">
                <span class="price-role-label" [style.color]="roleColor(role)">{{ role }}</span>
                @for (tier of allTiers; track tier) {
                  @if (s.priceIndex[role]?.[tier] !== undefined) {
                    <span
                      class="price-chip"
                      [style.border-color]="tierColor(tier)"
                      [style.color]="tierColor(tier)"
                    >
                      {{ tier.charAt(0) }}&thinsp;{{ s.priceIndex[role]![tier]! | number: '1.2-2' }}
                    </span>
                  }
                }
              </div>
            }
          </div>
        }

        <div class="auction-body">
          <!-- ── Left: Participants ──────────────────────── -->
          <aside class="participants-panel">
            <div class="panel-head">
              <p class="panel-heading">Chi sta giocando</p>
              <p class="panel-subheading">Soldi rimasti e quanto è completa la rosa</p>
            </div>

            @if (summaryLoading() && !summary()) {
              @for (_ of [1, 2, 3, 4, 5, 6, 7, 8]; track $index) {
                <app-skeleton height="72px" />
              }
            }

            @if (summary(); as s) {
              @for (p of s.participants; track p.participantId) {
                <div class="participant-card">
                  <div class="participant-header">
                    <span class="participant-name">{{ p.displayName }}</span>
                    <span class="participant-budget" [style.color]="budgetColor(p)">
                      {{ p.budgetResidual }} cr.
                    </span>
                  </div>
                  <div class="budget-bar">
                    <div
                      class="budget-bar-fill"
                      [style.width]="budgetPercent(p) + '%'"
                      [style.background]="budgetColor(p)"
                    ></div>
                  </div>
                  <div class="role-chips">
                    @for (role of rolesInBreakdown(p); track role) {
                      <span
                        class="role-chip"
                        [style.color]="roleColor(role)"
                        [style.border-color]="roleColor(role)"
                      >
                        {{ role }}&thinsp;{{ p.roleBreakdown[role] }}
                      </span>
                    }
                    @if (p.squad.length === 0) {
                      <span class="empty-squad" title="Nessun giocatore ancora acquistato">—</span>
                    }
                  </div>
                  @if (completionPct(p.participantId); as cp) {
                    <div class="completion-row" [title]="'Probabilità stimata di completare la rosa con il budget residuo'">
                      <span class="completion-label">Completamento rosa</span>
                      <span class="completion-value" [style.color]="completionColor(cp)">{{ cp | number: '1.0-0' }}%</span>
                      <div class="completion-bar">
                        <div class="completion-bar-fill" [style.width]="cp + '%'" [style.background]="completionColor(cp)"></div>
                      </div>
                    </div>
                  }
                  @if (mantraCoverageFor(p.participantId); as mods) {
                    <div class="mantra-coverage-row" aria-label="Schierabilità moduli Mantra">
                      <span class="completion-label">Moduli Mantra</span>
                      <div class="mantra-coverage-chips">
                        @for (m of mods; track m.label) {
                          <span class="formation-chip" [class.ok]="m.feasible" [class.ko]="!m.feasible"
                                [title]="m.feasible ? (m.label + ': ok') : (m.label + ': ' + deficitHint(m))">
                            {{ m.label }} {{ m.feasible ? '✓' : '✗' }}
                          </span>
                        }
                      </div>
                    </div>
                  }
                </div>
              }
            }
          </aside>

          <!-- ── Right: Main area ──────────────────────── -->
          <main class="auction-main">
            <div class="action-row">
              <!-- Lookup card -->
              <div class="card card--action">
                <div class="card-head">
                  <p class="card-section-label">1 · Cerca il giocatore</p>
                  <p class="card-section-hint">Ti diciamo subito quanto spendere e chi prendere al posto suo</p>
                </div>
                <div class="field-row field-row--compact">
                  <div class="field-group" style="flex:1">
                    <label class="field-label" for="altStrategy">Strategia (cap prezzo)</label>
                    <select
                      id="altStrategy"
                      class="field-input"
                      [ngModel]="altStrategyName"
                      (ngModelChange)="onStrategyChange($event)"
                    >
                      <option [ngValue]="null">Nessuna</option>
                      <option value="BALANCED">BALANCED</option>
                      <option value="SUPER_DEFENSIVE">SUPER_DEFENSIVE</option>
                      <option value="SUPER_OFFENSIVE">SUPER_OFFENSIVE</option>
                      <option value="MIXED">MIXED</option>
                    </select>
                  </div>
                </div>
                <div class="pool-autocomplete">
                  <div class="lookup-row">
                    <input
                      id="lookupQuery"
                      class="field-input"
                      placeholder="Cerca giocatore per nome, squadra o ruolo…"
                      [ngModel]="lookupQuery"
                      (ngModelChange)="lookupQuery = $event; onPoolQueryChange($event)"
                      (keydown.escape)="poolOpen.set(false)"
                      (keydown.enter)="lookupPlayer()"
                      [attr.aria-describedby]="'legend-lookupQuery'"
                      autocomplete="off"
                    />
                    @if (lookupLoading()) {
                      <span
                        class="spinner-sm"
                        style="flex-shrink:0;color:var(--color-accent)"
                        aria-label="Caricamento suggerimenti"
                      ></span>
                    }
                  </div>
                  <app-field-legend
                    fieldId="legend-lookupQuery"
                    [description]="LIVE_LEGENDS['lookupQuery'].description"
                    [examples]="LIVE_LEGENDS['lookupQuery'].examples"
                  />
                  @if (poolOpen() && poolSuggestions().length) {
                    <ul class="pool-dropdown" role="listbox">
                      @for (p of poolSuggestions(); track p.playerId) {
                        <li class="pool-option" role="option" (mousedown)="selectPoolPlayer(p)">
                          <span class="pool-name">
                            {{ p.name }}
                            @if (p.sampleCohort === 'LIMITED' || p.sampleCohort === 'INSUFFICIENT') {
                              <span class="ml-noisy-badge"
                                    [attr.title]="p.sampleCohort === 'INSUFFICIENT'
                                      ? 'Campione insufficiente (&lt;100 min)'
                                      : 'Campione limitato (100–799 min)'">
                                ⚠️ {{ p.sampleCohort === 'INSUFFICIENT' ? 'Insuff.' : 'Limited' }}
                              </span>
                            }
                          </span>
                          <span class="pool-meta">
                            <span
                              class="role-badge"
                              [style.color]="roleColor(p.role)"
                              [style.border-color]="roleColor(p.role)"
                              >{{ roleLabel(p) }}</span
                            >
                            {{ p.realTeam }} · quotazione {{ p.cost }} cr.
                          </span>
                        </li>
                      }
                    </ul>
                  }
                </div>

                @if (lookupError()) {
                  <p class="inline-error" role="alert">{{ lookupError() }}</p>
                }

                @if (projection(); as proj) {
                  <div
                    class="projection-row"
                    title="Prezzo atteso EWMA per il giocatore: è la stima di quanto dovrebbe essere aggiudicato in questo momento dell'asta."
                  >
                    <span class="proj-label">Prezzo atteso</span>
                    <span class="proj-price">{{ proj.expectedPrice | number: '1.0-0' }} cr.</span>
                    <span
                      class="tier-badge"
                      [style.color]="tierColor(proj.tier)"
                      [style.border-color]="tierColor(proj.tier)"
                      >{{ proj.tier }}</span
                    >
                  </div>
                }

                @if (altResult(); as alt) {
                  <div class="alternatives-grid">
                    @if (alt.lowCostAlternative; as lc) {
                      <div class="alt-card">
                        <p class="alt-label">Alternativa economica</p>
                        <p class="alt-name">{{ lc.name }}</p>
                        <p class="alt-meta">
                          {{ lc.realTeam }} · {{ roleLabel(lc) }} · {{ lc.cost }} cr.
                        </p>
                      </div>
                    }
                    @if (alt.closestAlternative; as cl) {
                      <div class="alt-card">
                        <p class="alt-label">Alternativa più simile</p>
                        <p class="alt-name">{{ cl.name }}</p>
                        <p class="alt-meta">
                          {{ cl.realTeam }} · {{ roleLabel(cl) }} · {{ cl.cost }} cr.
                        </p>
                      </div>
                    }
                    @if (alt.diversifiedAlternatives?.length) {
                      @for (d of alt.diversifiedAlternatives!; track d.playerId) {
                        <div class="alt-card alt-card--pareto">
                          <p class="alt-label">Pareto</p>
                          <p class="alt-name">{{ d.name }}</p>
                          <p class="alt-meta">
                            {{ d.realTeam }} · {{ roleLabel(d) }} · {{ d.cost }} cr.
                          </p>
                        </div>
                      }
                    }
                    @if (!alt.lowCostAlternative && !alt.closestAlternative && alt.reasonIfNone) {
                      <p class="alt-none">{{ alt.reasonIfNone }}</p>
                    }
                  </div>
                  @if (alt.maxAffordableBid != null || alt.strategyPriceCap != null) {
                    <div class="bid-caps-row">
                      @if (alt.maxAffordableBid != null) {
                        <span class="bid-cap" title="Max bid rispettando la riserva crediti">
                          Max bid: <strong>{{ alt.maxAffordableBid }}</strong> cr.
                        </span>
                      }
                      @if (alt.strategyPriceCap != null) {
                        <span class="bid-cap" title="Soglia strategia-aware sul prezzo atteso">
                          Cap strategia: <strong>{{ alt.strategyPriceCap }}</strong> cr.
                        </span>
                      }
                    </div>
                  }
                }
              </div>

              <!-- Record card -->
              <div class="card card--action">
                <div class="card-head">
                  <p class="card-section-label">2 · Registra l’acquisto</p>
                  <p class="card-section-hint">Chi ha vinto, a quanto, e in quale ruolo</p>
                </div>

                <div class="field-group">
                  <label class="field-label" for="recordPlayer"
                    >Giocatore <span class="field-hint">dal lookup</span></label
                  >
                  <input
                    id="recordPlayer"
                    class="field-input"
                    [ngModel]="recordPlayerName || recordPlayerId"
                    readonly
                    placeholder="seleziona dal Lookup →"
                    [style.color]="
                      recordPlayerId ? 'var(--color-text-primary)' : 'var(--color-text-secondary)'
                    "
                    [attr.aria-describedby]="'legend-recordPlayer'"
                  />
                  <app-field-legend
                    fieldId="legend-recordPlayer"
                    [description]="LIVE_LEGENDS['recordPlayer'].description"
                    [examples]="LIVE_LEGENDS['recordPlayer'].examples"
                  />
                </div>

                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="recordWinner"
                      >Vincitore</label
                    >
                    <select
                      id="recordWinner"
                      class="field-input"
                      [ngModel]="recordWinnerId"
                      (ngModelChange)="onWinnerChange($event)"
                      [attr.aria-describedby]="'legend-recordWinner'"
                    >
                      <option value="">— seleziona —</option>
                      @if (summary(); as s) {
                        @for (p of s.participants; track p.participantId) {
                          <option [value]="p.participantId">{{ p.displayName }}</option>
                        }
                      }
                    </select>
                    <app-field-legend
                      fieldId="legend-recordWinner"
                      [description]="LIVE_LEGENDS['recordWinner'].description"
                      [examples]="LIVE_LEGENDS['recordWinner'].examples"
                    />
                  </div>
                  @if (recordEligibleSlots.length > 1) {
                    <div class="field-group">
                      <label class="field-label" for="recordSlot"
                        >Slot MANTRA</label
                      >
                      <select
                        id="recordSlot"
                        class="field-input"
                        [(ngModel)]="recordAssignedSlot"
                      >
                        <option [ngValue]="null">Auto</option>
                        @for (slot of recordEligibleSlots; track slot) {
                          <option [ngValue]="slot">{{ slot }}</option>
                        }
                      </select>
                    </div>
                  }
                </div>

                <div class="field-group">
                  <label class="field-label" for="recordPrice"
                    >Prezzo pagato <span class="field-hint">cr. · aggiorna EWMA</span></label
                  >
                  <input
                    id="recordPrice"
                    class="field-input"
                    type="number"
                    min="1"
                    [(ngModel)]="recordPrice"
                    [attr.aria-describedby]="'legend-recordPrice'"
                  />
                  <app-field-legend
                    fieldId="legend-recordPrice"
                    [description]="LIVE_LEGENDS['recordPrice'].description"
                    [examples]="LIVE_LEGENDS['recordPrice'].examples"
                  />
                </div>

                @if (recordError()) {
                  <div class="inline-rejection" role="alert">
                    @if (recordRejectionCode()) {
                      <code class="rejection-code">{{ recordRejectionCode() }}</code>
                    }
                    <p class="rejection-msg">{{ recordError() }}</p>
                  </div>
                }

                <div class="record-actions">
                  <button
                    class="run-btn"
                    (click)="submitRecord()"
                    [disabled]="
                      recordLoading() || !recordPlayerId || !recordWinnerId || recordPrice < 1
                    "
                    title="Conferma l'assegnazione: scalo il prezzo dal budget del vincitore, aggiorno l'indice EWMA del ruolo/tier e aggiungo l'operazione allo storico."
                  >
                    @if (recordLoading()) {
                      <span class="spinner"></span> Registrazione…
                    } @else {
                      Registra assegnazione
                    }
                  </button>
                  <button
                    class="secondary-btn"
                    (click)="undoLast()"
                    [disabled]="undoLoading()"
                    title="Annulla l'ultima assegnazione registrata: ripristina il budget del vincitore e ripristina l'indice EWMA al valore precedente."
                  >
                    @if (undoLoading()) {
                      <span class="spinner-sm"></span>
                    } @else {
                      Annulla ultima
                    }
                  </button>
                </div>
              </div>
            </div>

            <!-- Assignment history -->
            @if (reversedAssignments().length) {
              <div class="card history-card">
                <p class="card-section-label">
                  Storico assegnazioni ({{ reversedAssignments().length }})
                </p>
                <div class="history-table-wrap">
                  <table class="squad-table">
                    <thead>
                      <tr>
                        <th title="Numero progressivo dell'asta nella sessione">#</th>
                        <th>Giocatore</th>
                        <th>Aggiudicatario</th>
                        <th title="Ruolo del giocatore (P/D/C/A)">Ruolo</th>
                        <th class="num" title="Prezzo finale di aggiudicazione in crediti">
                          Prezzo
                        </th>
                        <th title="Tier di quotazione al momento dell'asta">Tier</th>
                        <th
                          class="num"
                          title="Variazione dell'indice EWMA del ruolo/tier dopo l'assegnazione"
                        >
                          Δ Indice
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (a of reversedAssignments(); track a.sequenceNumber) {
                        <tr class="clickable-row"
                            (click)="openAssignmentPlayer(a)"
                            (keydown.enter)="openAssignmentPlayer(a)"
                            (keydown.space)="$event.preventDefault(); openAssignmentPlayer(a)"
                            tabindex="0"
                            role="button"
                            [attr.aria-label]="'Dettaglio ' + a.player.name">
                          <td class="seq">{{ a.sequenceNumber }}</td>
                          <td>
                            <p class="player-name">
                              {{ a.player.name }}
                              @if (a.player.sampleCohort === 'LIMITED' || a.player.sampleCohort === 'INSUFFICIENT') {
                                <span class="ml-noisy-badge">⚠️ {{ a.player.sampleCohort === 'INSUFFICIENT' ? 'Insuff.' : 'Limited' }}</span>
                              }
                            </p>
                            <p class="team-name">{{ a.player.realTeam }}</p>
                          </td>
                          <td>{{ winnerName(a.winnerParticipantId) }}</td>
                          <td>
                            <span
                              class="role-badge"
                              [style.color]="roleColor(a.role)"
                              [style.border-color]="roleColor(a.role)"
                              >{{ a.assignedSlot || a.role }}</span
                            >
                          </td>
                          <td class="num accent">{{ a.finalPrice }}</td>
                          <td>
                            <span
                              class="tier-badge"
                              [style.color]="tierColor(a.tier)"
                              [style.border-color]="tierColor(a.tier)"
                              >{{ a.tier }}</span
                            >
                          </td>
                          <td class="num faded">
                            {{ a.priceIndexAfter - a.priceIndexBefore | number: '+1.3-3' }}
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            } @else if (summary()) {
              <div class="card empty-history">
                <p>
                  Nessuna assegnazione registrata. Cerca un giocatore nel Lookup, poi registra la
                  sua vendita.
                </p>
              </div>
            }

            <!-- VAR/ESV ranking -->
            @if (varRanking().length || varLoading()) {
              <div class="card">
                <p class="card-section-label">
                  Ranking VAR/ESV (surplus vs expected_price) disponibili (ESV vs prezzo EWMA)
                  @if (varLoading()) {
                    <span class="spinner-sm" style="margin-left:6px"></span>
                  }
                </p>
                <app-field-legend
                  fieldId="legend-varRanking"
                  [description]="LIVE_LEGENDS['varRanking'].description"
                  [examples]="LIVE_LEGENDS['varRanking'].examples"
                />
                @if (varRanking().length) {
                  <div class="history-table-wrap">
                    <table class="squad-table">
                      <thead>
                        <tr>
                          <th
                            class="sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('name')"
                            (click)="cycleVarSort('name')"
                            (keydown.enter)="cycleVarSort('name')"
                            (keydown.space)="$event.preventDefault(); cycleVarSort('name')"
                            title="Ordina per nome del giocatore"
                          >
                            <span>Giocatore</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('name')
                            }}</span>
                          </th>
                          <th
                            class="sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('role')"
                            (click)="cycleVarSort('role')"
                            (keydown.enter)="cycleVarSort('role')"
                            (keydown.space)="$event.preventDefault(); cycleVarSort('role')"
                            title="Ordina per ruolo (P/D/C/A)"
                          >
                            <span>Ruolo</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('role')
                            }}</span>
                          </th>
                          <th
                            class="num sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('esv')"
                            (click)="cycleVarSort('esv')"
                            (keydown.enter)="cycleVarSort('esv')"
                            (keydown.space)="$event.preventDefault(); cycleVarSort('esv')"
                            title="Ordina per Expected Season Value"
                          >
                            <span>ESV</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('esv')
                            }}</span>
                          </th>
                          <th
                            class="num sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('expectedPrice')"
                            (click)="cycleVarSort('expectedPrice')"
                            (keydown.enter)="cycleVarSort('expectedPrice')"
                            (keydown.space)="$event.preventDefault(); cycleVarSort('expectedPrice')"
                            title="Ordina per prezzo atteso EWMA"
                          >
                            <span>Prezzo atteso</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('expectedPrice')
                            }}</span>
                          </th>
                          <th
                            class="num sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('seasonValue')"
                            (click)="cycleVarSort('seasonValue')"
                            (keydown.enter)="cycleVarSort('seasonValue')"
                            (keydown.space)="$event.preventDefault(); cycleVarSort('seasonValue')"
                            title="Ordina per valore di stagione"
                          >
                            <span>Val. stagione</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('seasonValue')
                            }}</span>
                          </th>
                          <th
                            class="num sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('startProbability')"
                            (click)="cycleVarSort('startProbability')"
                            (keydown.enter)="cycleVarSort('startProbability')"
                            (keydown.space)="
                              $event.preventDefault(); cycleVarSort('startProbability')
                            "
                            title="Ordina per probabilità di titolarità"
                          >
                            <span>% Titolarità</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('startProbability')
                            }}</span>
                          </th>
                          <th
                            class="sortable"
                            tabindex="0"
                            role="columnheader"
                            [attr.aria-sort]="varAriaSort('buySignal')"
                            (click)="cycleVarSort('buySignal')"
                            (keydown.enter)="cycleVarSort('buySignal')"
                            (keydown.space)="$event.preventDefault(); cycleVarSort('buySignal')"
                            title="Ordina per segnale di acquisto (COMPRA prima)"
                          >
                            <span>Segnale</span>
                            <span class="sort-indicator" aria-hidden="true">{{
                              varSortIndicator('buySignal')
                            }}</span>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        @for (v of sortedVarRanking(); track v.playerId) {
                          <tr class="clickable-row"
                              (click)="openVarPlayer(v)"
                              (keydown.enter)="openVarPlayer(v)"
                              (keydown.space)="$event.preventDefault(); openVarPlayer(v)"
                              tabindex="0"
                              role="button"
                              [attr.aria-label]="'Dettaglio ' + v.name">
                            <td>
                              {{ v.name }}
                              @if (v.sampleCohort === 'LIMITED' || v.sampleCohort === 'INSUFFICIENT') {
                                <span class="ml-noisy-badge"
                                      [attr.title]="v.sampleCohort === 'INSUFFICIENT'
                                        ? 'Campione insufficiente (&lt;100 min)'
                                        : 'Campione limitato (100–799 min)'">
                                  ⚠️ {{ v.sampleCohort === 'INSUFFICIENT' ? 'Insuff.' : 'Limited' }}
                                </span>
                              }
                            </td>
                            <td>
                              <span
                                class="role-badge"
                                [style.color]="roleColor(v.role)"
                                [style.border-color]="roleColor(v.role)"
                                >{{ v.role }}</span
                              >
                            </td>
                            <td
                              class="num"
                              [style.color]="
                                v.esv > 0
                                  ? 'var(--color-success, #22C55E)'
                                  : 'var(--color-text-secondary)'
                              "
                            >
                              {{ v.esv | number: '1.1-1' }}
                            </td>
                            <td class="num faded">{{ v.expectedPrice | number: '1.0-0' }}</td>
                            <td class="num faded">
                              {{ v.seasonValue != null ? (v.seasonValue | number: '1.1-1') : '—' }}
                            </td>
                            <td class="num faded">
                              {{
                                v.startProbability != null
                                  ? (v.startProbability * 100 | number: '1.0-0') + '%'
                                  : '—'
                              }}
                            </td>
                            <td>
                              @if (v.buySignal) {
                                <span class="esv-badge esv-buy" title="Affare: ESV positivo"
                                  >COMPRA</span
                                >
                              } @else {
                                <span
                                  class="esv-badge esv-hold"
                                  title="EVITARE: ESV negativo o nullo"
                                  >-</span
                                >
                              }
                            </td>
                          </tr>
                        }
                      </tbody>
                    </table>
                  </div>
                }
              </div>
            }
          </main>
        </div>
      </div>
    } @else {
      <!-- ═══════════════════════ SETUP VIEW ══════════════════════ -->
      <div class="auction-page">
        <header class="page-header">
          <div>
            <h1 class="page-title">Tracker Asta</h1>
            <p class="page-subtitle">
              Prepara l’asta in 2 minuti. Le opzioni tecniche restano nascoste finché non ti servono.
            </p>
          </div>
        </header>

        <div class="setup-body">
          <!-- Config panel -->
          <aside class="config-panel card">

            <p class="section-divider">Pool e sessione</p>

            <div class="field-group">
              <label class="field-label" for="auction-ruleset">Regolamento</label>
              <select
                id="auction-ruleset"
                class="field-input"
                [ngModel]="ruleset"
                (ngModelChange)="onRulesetChange($event)"
                [attr.aria-describedby]="'legend-auction-ruleset'"
              >
                <option value="CLASSIC">CLASSIC — 4 ruoli (P/D/C/A)</option>
                <option value="MANTRA">MANTRA — 12 ruoli multi-slot</option>
              </select>
              <app-field-legend
                fieldId="legend-auction-ruleset"
                description="Cambia il modello di assegnazione ruoli. CLASSIC: 4 ruoli (P/D/C/A) con quote 3/8/8/6. MANTRA: 12 ruoli multi-slot (eligible_roles) e quote Por/Dc/B/… che devono sommare a 25. Stesso selettore già in produzione sull'optimizer."
                [examples]="[
                  { label: 'CLASSIC', value: 'comportamento storico, zero regressioni' },
                  { label: 'MANTRA', value: 'ruoli modulari e multi-ruolo' }
                ]"
              />
            </div>

            <div class="field-group">
              <label class="field-label" for="seasonStart">Stagione del pool (listini + predizioni ML)</label>
              @if (seasonsLoading()) {
                <app-skeleton height="36px" />
              } @else {
                <select
                  id="seasonStart"
                  class="field-input"
                  [(ngModel)]="seasonStart"
                  [attr.aria-describedby]="'legend-seasonStart'"
                >
                  @for (s of seasons(); track s) {
                    <option [value]="s">{{ s }}/{{ s + 1 }}</option>
                  }
                </select>
              }
              <app-field-legend
                fieldId="legend-seasonStart"
                [description]="SETUP_LEGENDS['seasonStart'].description"
                [examples]="SETUP_LEGENDS['seasonStart'].examples"
              />
            </div>
            
            <p class="section-divider">Profilo strategico</p>

            <div class="field-group">
              <label class="field-label" for="auction-preset">Preset d'asta (precompila EWMA, inflazione, alternative)</label>
              <select
                id="auction-preset"
                class="field-input"
                [ngModel]="selectedPresetId"
                (ngModelChange)="onPresetChange($event)"
                [attr.aria-describedby]="'legend-auction-preset'"
              >
                <option [ngValue]="AUCTION_PRESET_NONE">Personalizzato (nessun preset)</option>
                @for (p of presets; track p.id) {
                  <option [ngValue]="p.id">{{ p.labelIt }} — {{ p.name }}</option>
                }
              </select>
              @if (activePreset; as preset) {
                <p class="preset-description" id="legend-auction-preset">{{ preset.description }}</p>
              } @else {
                <p class="preset-description muted" id="legend-auction-preset">
                  Scegli un profilo per precompilare drift EWMA, inflazione, alternative e valuation.
                  Stagione e partecipanti restano sotto il tuo controllo.
                </p>
              }
            </div>

            

            <p class="section-divider">Partecipanti e crediti</p>

            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="numParticipants"
                  >Partecipanti asta <span class="field-hint">= len(participants) all'init</span></label
                >
                <input
                  id="numParticipants"
                  class="field-input"
                  type="number"
                  min="2"
                  max="20"
                  [(ngModel)]="numParticipants"
                  (change)="resizeParticipants()"
                  [attr.aria-describedby]="'legend-numParticipants'"
                />
                <app-field-legend
                  fieldId="legend-numParticipants"
                  [description]="SETUP_LEGENDS['numParticipants'].description"
                  [examples]="SETUP_LEGENDS['numParticipants'].examples"
                />
              </div>
              <div class="field-group">
                <label class="field-label" for="defaultBudget"
                  >Budget crediti per manager <span class="field-hint">propaga a participants[]</span></label
                >
                <input
                  id="defaultBudget"
                  class="field-input"
                  type="number"
                  min="100"
                  max="2000"
                  step="25"
                  [ngModel]="defaultBudget"
                  (ngModelChange)="defaultBudget = +$event; applyDefaultBudget()"
                  [attr.aria-describedby]="'legend-defaultBudget'"
                />
                <app-field-legend
                  fieldId="legend-defaultBudget"
                  [description]="SETUP_LEGENDS['defaultBudget'].description"
                  [examples]="SETUP_LEGENDS['defaultBudget'].examples"
                />
              </div>
            </div>

            <p class="section-divider">Quote rosa (vincolo hard per manager)</p>

            <div class="quota-grid">
              @for (role of quotaRoles; track role) {
                <div class="field-group">
                  <label
                    class="field-label"
                    [style.color]="roleColor(role)"
                    [attr.for]="'role-' + role"
                  >
                    @switch (role) {
                      @case ('P') {
                        Portieri (P) — max slot per manager
                      }
                      @case ('D') {
                        Difensori (D) — max slot per manager
                      }
                      @case ('C') {
                        Centrocampisti (C) — max slot per manager
                      }
                      @case ('A') {
                        Attaccanti (A) — max slot per manager
                      }
                      @default {
                        {{ role }} — max slot per manager
                      }
                    }
                  </label>
                  <input
                    [id]="'role-' + role"
                    class="field-input"
                    type="number"
                    min="1"
                    max="20"
                    [ngModel]="roleQuotas[role] ?? 0"
                    (ngModelChange)="roleQuotas[role] = +$event"
                    [attr.aria-describedby]="'legend-roleQuotas'"
                  />
                </div>
              }
            </div>
            <app-field-legend
              fieldId="legend-roleQuotas"
              [description]="SETUP_LEGENDS['roleQuotas'].description"
              [examples]="SETUP_LEGENDS['roleQuotas'].examples"
            />

            <p class="section-divider">Baseline cost (inflazione statica sul listino)</p>

            <label class="strategy-check" [class.active]="useInflationBaseline">
              <input type="checkbox" [(ngModel)]="useInflationBaseline" />
              <span>Usa estimate_effective_cost come baseline EWMA (inflazione listino)</span>
            </label>

            @if (useInflationBaseline) {
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="inflationPct"
                    >
                    >Soglia percentile: sotto = baseline = listino scalato <span class="field-hint">0–1</span></label
                  >
                  <input
                    id="inflationPct"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    [ngModel]="inflationPercentileThreshold"
                    (ngModelChange)="inflationPercentileThreshold = +$event"
                    [attr.aria-describedby]="'legend-inflationPercentileThreshold'"
                  />
                  <app-field-legend
                    fieldId="legend-inflationPercentileThreshold"
                    [description]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].description"
                    [examples]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].examples"
                  />
                </div>
                <div class="field-group">
                  <label class="field-label" for="maxInflation"
                    >
                    >Cap costo effettivo / listino <span class="field-hint">≥ 1.0</span></label
                  >
                  <input
                    id="maxInflation"
                    class="field-input"
                    type="number"
                    min="1"
                    max="5"
                    step="0.05"
                    [ngModel]="maxInflationMultiplier"
                    (ngModelChange)="maxInflationMultiplier = +$event"
                    [attr.aria-describedby]="'legend-maxInflationMultiplier'"
                  />
                  <app-field-legend
                    fieldId="legend-maxInflationMultiplier"
                    [description]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].description"
                    [examples]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].examples"
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="baseInflationRate"
                    >
                    >Tasso base inflazione <span class="field-hint">partecipanti extra</span></label
                  >
                  <input
                    id="baseInflationRate"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    [ngModel]="baseInflationRate"
                    (ngModelChange)="baseInflationRate = +$event"
                    [attr.aria-describedby]="'legend-baseInflationRate'"
                  />
                  <app-field-legend
                    fieldId="legend-baseInflationRate"
                    [description]="OPTIMIZER_LEGENDS['baseInflationRate'].description"
                    [examples]="OPTIMIZER_LEGENDS['baseInflationRate'].examples"
                  />
                </div>
                <div class="field-group">
                  <label class="field-label" for="baselineParticipants"
                    >
                    >Baseline partecipanti <span class="field-hint">oltre → extra inflazione</span></label
                  >
                  <input
                    id="baselineParticipants"
                    class="field-input"
                    type="number"
                    min="2"
                    max="20"
                    step="1"
                    [ngModel]="baselineParticipants"
                    (ngModelChange)="baselineParticipants = +$event"
                    [attr.aria-describedby]="'legend-baselineParticipants'"
                  />
                  <app-field-legend
                    fieldId="legend-baselineParticipants"
                    [description]="OPTIMIZER_LEGENDS['baselineParticipants'].description"
                    [examples]="OPTIMIZER_LEGENDS['baselineParticipants'].examples"
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="teamStrengthMul"
                    >
                    >Peso Elo club sul baseline_cost <span class="field-hint">0 = off</span></label
                  >
                  <input
                    id="teamStrengthMul"
                    class="field-input"
                    type="number"
                    min="0"
                    max="2"
                    step="0.05"
                    [ngModel]="teamStrengthMultiplier"
                    (ngModelChange)="teamStrengthMultiplier = +$event"
                    [attr.aria-describedby]="'legend-teamStrengthMultiplier'"
                  />
                  <app-field-legend
                    fieldId="legend-teamStrengthMultiplier"
                    [description]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].description"
                    [examples]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].examples"
                  />
                </div>
              </div>
            }

            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="referenceBudget"
                  >
                  >Reference budget listino <span class="field-hint">scala qt_a</span></label
                >
                <input
                  id="referenceBudget"
                  class="field-input"
                  type="number"
                  min="1"
                  max="10000"
                  step="50"
                  [(ngModel)]="referenceBudget"
                  [attr.aria-describedby]="'legend-referenceBudget'"
                />
                <app-field-legend
                  fieldId="legend-referenceBudget"
                  [description]="SETUP_LEGENDS['referenceBudget'].description"
                  [examples]="SETUP_LEGENDS['referenceBudget'].examples"
                />
              </div>
              <div class="field-group">
                <label class="field-label" for="budgetInitial"
                  >
                  >Budget sessione (scala listino) <span class="field-hint">cr.</span></label
                >
                <input
                  id="budgetInitial"
                  class="field-input"
                  type="number"
                  min="1"
                  max="10000"
                  step="50"
                  [(ngModel)]="budgetInitial"
                  [attr.aria-describedby]="'legend-budgetInitial'"
                />
                <app-field-legend
                  fieldId="legend-budgetInitial"
                  [description]="SETUP_LEGENDS['budgetInitial'].description"
                  [examples]="SETUP_LEGENDS['budgetInitial'].examples"
                />
              </div>
            </div>

            <p class="section-divider">Metrica ranking VAR/ESV</p>

            <div class="field-group">
              <label class="field-label" for="valuationMode"
                >
                >Metrica del ranking VAR/ESV</label
              >
              <select
                id="valuationMode"
                class="field-input"
                [(ngModel)]="valuationMode"
                [attr.aria-describedby]="'legend-valuationMode'"
              >
                <option value="PER_MATCH_RATING">Per-Match Rating (media voto ponderata)</option>
                <option value="SEASON_VALUE">Season Value (valore complessivo di stagione)</option>
              </select>
              <app-field-legend
                fieldId="legend-valuationMode"
                [description]="SETUP_LEGENDS['valuationMode'].description"
                [examples]="SETUP_LEGENDS['valuationMode'].examples"
              />
            </div>

            <div class="field-group">
              <label class="field-label" for="hybridBlend"
                >Blend fpIbrido (hybrid) <span class="field-hint">0–1, 0 = off</span></label
              >
              <input
                id="hybridBlend"
                class="field-input"
                type="number"
                min="0"
                max="1"
                step="0.05"
                [(ngModel)]="hybridBlend"
              />
              <app-field-legend
                fieldId="legend-hybridBlend"
                description="Peso del segnale MANTRA-ibrido (fpIbrido) nel ranking VAR/ESV. 0 = disattivato. Stesso pattern dell'optimizer hybridBlend."
                [examples]="[
                  { label: '0', value: 'solo projected_score / season_value' },
                  { label: '0.3–0.5', value: 'blend moderato' },
                  { label: '1', value: 'solo fpIbrido dove disponibile' }
                ]"
              />
            </div>

            <p class="section-divider">Filtro ranking VAR e replacement level</p>

            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="replacementMethod"
                  >
                  >Metodo replacement level <span class="field-hint">solo VAR/ESV</span></label
                >
                <select
                  id="replacementMethod"
                  class="field-input"
                  [(ngModel)]="replacementMethod"
                  [attr.aria-describedby]="'legend-replacementMethod'"
                >
                  <option value="percentile">Percentile (bottom-N% per ruolo)</option>
                  <option value="roster_depth">Roster depth (quota rosa per ruolo)</option>
                </select>
                <app-field-legend
                  fieldId="legend-replacementMethod"
                  [description]="OPTIMIZER_LEGENDS['replacementMethod'].description"
                  [examples]="OPTIMIZER_LEGENDS['replacementMethod'].examples"
                />
              </div>
              <div class="field-group">
                <label class="field-label" for="minStartProb"
                  >
                  >Filtro start_probability sul ranking VAR <span class="field-hint">non sul pool</span></label
                >
                <input
                  id="minStartProb"
                  class="field-input"
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  [ngModel]="minStartProbability"
                  (ngModelChange)="
                    minStartProbability = $event === null || $event === '' ? null : +$event
                  "
                  [attr.aria-describedby]="'legend-minStartProbability'"
                />
                <app-field-legend
                  fieldId="legend-minStartProbability"
                  [description]="OPTIMIZER_LEGENDS['minStartProbability'].description"
                  [examples]="OPTIMIZER_LEGENDS['minStartProbability'].examples"
                />
              </div>
            </div>

            <!-- Advanced -->
            <button class="advanced-toggle" (click)="showAdvanced = !showAdvanced">
              {{ showAdvanced ? 'Nascondi opzioni tecniche' : 'Mostra opzioni tecniche (EWMA, spillover, tier…)' }}
            </button>

            @if (showAdvanced) {
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="alpha"
                    >
                    >EWMA alpha <span class="field-hint">index ← (1−α)·index + α·ratio</span></label
                  >
                  <input
                    id="alpha"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    [(ngModel)]="alpha"
                    [attr.aria-describedby]="'legend-alpha'"
                  />
                  <app-field-legend
                    fieldId="legend-alpha"
                    [description]="SETUP_LEGENDS['alpha'].description"
                    [examples]="SETUP_LEGENDS['alpha'].examples"
                  />
                </div>
                <div class="field-group">
                  <label class="field-label" for="spilloverAdj"
                    >
                    >Spillover tier adiacenti <span class="field-hint">stesso ruolo</span></label
                  >
                  <input
                    id="spilloverAdj"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    [(ngModel)]="spilloverAdj"
                    [attr.aria-describedby]="'legend-spilloverAdj'"
                  />
                  <app-field-legend
                    fieldId="legend-spilloverAdj"
                    [description]="SETUP_LEGENDS['spilloverAdj'].description"
                    [examples]="SETUP_LEGENDS['spilloverAdj'].examples"
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="spilloverCross"
                    >
                    >Spillover cross-ruolo <span class="field-hint">default 0 = off</span></label
                  >
                  <input
                    id="spilloverCross"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    [(ngModel)]="spilloverCross"
                    [attr.aria-describedby]="'legend-spilloverCross'"
                  />
                  <app-field-legend
                    fieldId="legend-spilloverCross"
                    [description]="SETUP_LEGENDS['spilloverCross'].description"
                    [examples]="SETUP_LEGENDS['spilloverCross'].examples"
                  />
                </div>
                <div class="field-group">
                  <label class="field-label" for="lowCostPercentile"
                    >
                    >Soglia alternative low-cost <span class="field-hint">percentile expected_price</span></label
                  >
                  <input
                    id="lowCostPercentile"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    [(ngModel)]="lowCostPercentile"
                    [attr.aria-describedby]="'legend-lowCostPercentile'"
                  />
                  <app-field-legend
                    fieldId="legend-lowCostPercentile"
                    [description]="SETUP_LEGENDS['lowCostPercentile'].description"
                    [examples]="SETUP_LEGENDS['lowCostPercentile'].examples"
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="minIndex"
                    >
                    >Floor indice EWMA <span class="field-hint">clamp minimo</span></label
                  >
                  <input
                    id="minIndex"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    [(ngModel)]="minIndex"
                    [attr.aria-describedby]="'legend-minIndex'"
                  />
                  <app-field-legend
                    fieldId="legend-minIndex"
                    [description]="SETUP_LEGENDS['minIndex'].description"
                    [examples]="SETUP_LEGENDS['minIndex'].examples"
                  />
                </div>
                <div class="field-group">
                  <label class="field-label" for="maxIndex"
                    >
                    >Cap indice EWMA <span class="field-hint">clamp massimo</span></label
                  >
                  <input
                    id="maxIndex"
                    class="field-input"
                    type="number"
                    min="1"
                    max="5"
                    step="0.1"
                    [(ngModel)]="maxIndex"
                    [attr.aria-describedby]="'legend-maxIndex'"
                  />
                  <app-field-legend
                    fieldId="legend-maxIndex"
                    [description]="SETUP_LEGENDS['maxIndex'].description"
                    [examples]="SETUP_LEGENDS['maxIndex'].examples"
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="tierLow"
                    >
                    >Soglia percentile → tier LOW <span class="field-hint">pct &lt; low</span></label
                  >
                  <input
                    id="tierLow"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    [(ngModel)]="tierLow"
                    [attr.aria-describedby]="'legend-tierLow'"
                  />
                  <app-field-legend
                    fieldId="legend-tierLow"
                    [description]="SETUP_LEGENDS['tierLow'].description"
                    [examples]="SETUP_LEGENDS['tierLow'].examples"
                  />
                </div>
                <div class="field-group">
                  <label class="field-label" for="tierTop"
                    >
                    >Soglia percentile → tier TOP <span class="field-hint">pct ≥ top</span></label
                  >
                  <input
                    id="tierTop"
                    class="field-input"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    [(ngModel)]="tierTop"
                    [attr.aria-describedby]="'legend-tierTop'"
                  />
                  <app-field-legend
                    fieldId="legend-tierTop"
                    [description]="SETUP_LEGENDS['tierTop'].description"
                    [examples]="SETUP_LEGENDS['tierTop'].examples"
                  />
                </div>
              </div>
            }

            @if (initError()) {
              <app-error-boundary
                title="Errore di inizializzazione sessione"
                [message]="initError()!"
              />
            }

            <button
              class="run-btn"
              (click)="startAuction()"
              [disabled]="starting() || !seasons().length"
            >
              @if (starting()) {
                <span class="spinner"></span> Avvio in corso…
              } @else {
                Avvia asta
              }
            </button>

            <button
              class="secondary-btn full-w"
              (click)="fileInput.click()"
              [disabled]="starting()"
              title="Carica un file JSON precedentemente esportato con 'Salva sessione' per riprendere un'asta interrotta."
            >
              Riprendi da file di salvataggio (.json)
            </button>
            <input
              #fileInput
              type="file"
              accept=".json"
              style="display:none"
              (change)="onResumeFile($event)"
            />
          </aside>

          <!-- Participants editor -->
          <section class="setup-right">
            <div class="card">
              <p class="card-section-label" style="margin-bottom:12px">
                Elenco partecipanti ({{ participants().length }}) — modifica nome e budget
                individuale
              </p>
              <div class="participants-list">
                <div class="participants-list-header">
                  <span>Nome visualizzato</span><span>Budget iniziale (cr.)</span>
                </div>
                @for (p of participants(); track p.participantId; let i = $index) {
                  <div class="participant-edit-row">
                    <input
                      class="field-input"
                      [ngModel]="p.displayName"
                      (ngModelChange)="updateName(i, $event)"
                      [placeholder]="'Squadra ' + (i + 1)"
                      [attr.aria-label]="'Nome partecipante ' + (i + 1)"
                    />
                    <input
                      class="field-input budget-input"
                      type="number"
                      [ngModel]="p.budgetInitial"
                      (ngModelChange)="updateBudget(i, +$event)"
                      min="100"
                      max="2000"
                      step="25"
                      [attr.aria-label]="'Budget iniziale partecipante ' + (i + 1)"
                    />
                  </div>
                }
              </div>
            </div>

            <app-auction-simulation
              [participants]="participants()"
              [config]="setupAuctionConfig"
              [seasonStart]="seasonStart"
            />
          </section>
        </div>
      </div>
    }

    @if (selectedPlayer(); as p) {
      <app-auction-player-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
  `,
  styleUrls: ['./auction.component.scss'],
})
export class AuctionComponent {
  private readonly auctionService = inject(AuctionService);
  private readonly quotationService = inject(QuotationService);

  readonly allRoles: readonly AuctionRole[] = AUCTION_ROLES;
  /** Roles shown in the quota grid; switches with ruleset. */
  get quotaRoles(): readonly string[] {
    return this.ruleset === 'MANTRA' ? MANTRA_ROLES : AUCTION_ROLES;
  }

  /**
   * Roles used in the live price-index strip and related displays.
   * Prefer keys present in the live summary priceIndex; fall back to quota roles.
   */
  displayRoles(): readonly string[] {
    const s = this.summary();
    if (s?.priceIndex && Object.keys(s.priceIndex).length) {
      return Object.keys(s.priceIndex);
    }
    return this.quotaRoles;
  }

  /** Roles currently filled for a participant (any ruleset). */
  rolesInBreakdown(p: AuctionParticipantState): string[] {
    return Object.keys(p.roleBreakdown ?? {}).filter((r) => (p.roleBreakdown[r] ?? 0) > 0);
  }

  roleLabel(p: { role: string; eligibleRoles?: string[] | null }): string {
    if (p.eligibleRoles && p.eligibleRoles.length > 1) {
      return p.eligibleRoles.join('/');
    }
    if (p.eligibleRoles && p.eligibleRoles.length === 1) {
      return p.eligibleRoles[0];
    }
    return p.role;
  }

  completionPct(participantId: string): number | null {
    const cp = this.summary()?.completionProbability?.[participantId];
    if (cp == null || Number.isNaN(cp)) return null;
    return Math.round(Math.max(0, Math.min(1, cp)) * 100);
  }

  completionColor(pct: number): string {
    if (pct >= 70) return 'var(--color-success, #22c55e)';
    if (pct >= 40) return 'var(--color-accent)';
    return 'var(--color-danger, #ef4444)';
  }

  /** Sorted list of Mantra module coverages for a participant, or null if absent. */
  mantraCoverageFor(participantId: string): MantraModuleCoverage[] | null {
    const map = this.summary()?.mantraModuleCoverage?.[participantId];
    if (!map) return null;
    return Object.values(map).sort((a, b) => a.label.localeCompare(b.label));
  }

  deficitHint(m: MantraModuleCoverage): string {
    const parts = Object.entries(m.deficits ?? {}).map(([k, v]) => `${k}−${v}`);
    return parts.length ? parts.join(', ') : 'non schierabile';
  }

  readonly allTiers: readonly AuctionTier[] = ['LOW', 'MID', 'TOP'];

  /** Legende dei campi del pannello di setup (configurazione iniziale). */
  protected readonly SETUP_LEGENDS = SETUP_LEGENDS;
  /** Legende dei campi della vista live (lookup + registrazione). */
  protected readonly LIVE_LEGENDS = LIVE_LEGENDS;

  protected readonly OPTIMIZER_LEGENDS = OPTIMIZER_LEGENDS;

  /** Preset catalog (immutable). Exposed for the setup select. */
  readonly allPresets: readonly AuctionPreset[] = AUCTION_PRESETS;

  /** Presets compatible with the currently selected ruleset. */
  get presets(): readonly AuctionPreset[] {
    return this.allPresets.filter(
      (p) =>
        p.rulesetTarget === 'BOTH' ||
        p.rulesetTarget === this.ruleset,
    );
  }
  protected readonly AUCTION_PRESET_NONE = AUCTION_PRESET_NONE;

  /**
   * Currently selected preset id. Empty string = operator-driven custom config.
   * Applying a preset patches setup form fields; it does not start the session.
   */
  selectedPresetId: string = AUCTION_PRESET_NONE;

  get activePreset(): AuctionPreset | undefined {
    return findAuctionPreset(this.selectedPresetId);
  }

  // ── Setup form state (plain properties — bound via (change) events) ──
  seasonStart = SEASON_FALLBACK_LIST[0];
  numParticipants = 8;
  defaultBudget = 500;
  showAdvanced = false;
  useInflationBaseline = true;
  referenceBudget = 300;
  budgetInitial = 300;
  ruleset: AuctionRuleset = 'CLASSIC';
  roleQuotas: Partial<Record<string, number>> = { P: 3, D: 8, C: 8, A: 6 };
  valuationMode: ValuationMode = 'PER_MATCH_RATING';
  /** WS3 #2: fpIbrido blend weight for VarEngine (0 = off). */
  hybridBlend = 0.0;
  /** Optional strategy for alternatives price cap (WS3 #5). */
  altStrategyName: string | null = null;

  /** Switch ruleset and reset quotas to the corresponding defaults. */
  onRulesetChange(value: AuctionRuleset): void {
    this.ruleset = value;
    if (value === 'MANTRA') {
      this.roleQuotas = { ...MANTRA_DEFAULT_QUOTAS };
    } else {
      this.roleQuotas = { P: 3, D: 8, C: 8, A: 6 };
    }
    // Drop selected preset if it is no longer compatible with the active ruleset.
    const selected = findAuctionPreset(this.selectedPresetId);
    if (
      selected &&
      selected.rulesetTarget !== 'BOTH' &&
      selected.rulesetTarget !== value
    ) {
      this.selectedPresetId = AUCTION_PRESET_NONE;
    }
  }

  // ── Inflation config (only sent when useInflationBaseline = true) ──
  // Defaults mirror the backend Pydantic InflationConfigSchema defaults
  // so the auction screen stays behaviorally equivalent to the optimizer
  // when the user enables the toggle without touching any slider.
  /** Soglia (0–1) di percentile per attivare la moltiplicazione. Default 0.7. */
  inflationPercentileThreshold = 0.7;
  /** Moltiplicatore massimo applicato al listino base. Default 1.6. */
  maxInflationMultiplier = 1.6;
  /** Tasso di inflazione di base applicato a tutti i prezzi. Default 0.05 (5%). */
  baseInflationRate = 0.05;
  /** Numero di partecipanti "baseline" del modello. Default 8. */
  baselineParticipants = 8;
  /** Peso aggiustamento Elo di Club sul costo. 0 = disattivato (default backend). */
  teamStrengthMultiplier = 0.0;

  // ── Replacement level & start-probability pre-filter ──
  /** Replacement level: percentile (default) o roster_depth. */
  replacementMethod: 'percentile' | 'roster_depth' = 'percentile';
  /** Soglia minima di start_probability; null = nessun filtro (default). */
  minStartProbability: number | null = null;

  alpha = 0.3;
  spilloverAdj = 0.1;
  spilloverCross = 0.05;
  lowCostPercentile = 0.3;
  minIndex = 0.5;
  maxIndex = 2.0;
  tierLow = 0.3;
  tierTop = 0.7;

  private readonly destroyRef = inject(DestroyRef);

  // ── Async signals ────────────────────────────────────────────────────
  readonly seasons = signal<number[]>([]);
  readonly seasonsLoading = signal(true);
  readonly participants = signal<AuctionParticipantSetup[]>(makeParticipants(8, 500));
  readonly starting = signal(false);
  readonly initError = signal<string | null>(null);
  /** Players excluded from server-built pool (missing projection). */
  readonly nExcludedNoProjection = signal(0);

  readonly sessionId = signal<string | null>(null);
  readonly summary = signal<AuctionSummary | null>(null);
  readonly summaryLoading = signal(false);

  readonly projection = signal<ProjectionResponse | null>(null);
  readonly altResult = signal<AlternativesResponse | null>(null);
  readonly lookupLoading = signal(false);
  readonly lookupError = signal<string | null>(null);

  readonly recordLoading = signal(false);
  readonly recordError = signal<string | null>(null);
  readonly recordRejectionCode = signal<string | null>(null);
  readonly undoLoading = signal(false);

  readonly selectedPlayer = signal<AuctionDrawerPlayer | null>(null);
    readonly varRanking = signal<VarRankingItem[]>([]);
  readonly varLoading = signal(false);

  /**
   * Ordinamento della tabella "Ranking VAR/ESV (surplus vs expected_price)".
   * Stato: `varSortKey=null` e `varSortDir=null` ⇒ ordine naturale del backend.
   * `cycleVarSort(key)` fa: null → asc → desc → null.
   */
  readonly varSortKey = signal<VarSortKey | null>(null);
  readonly varSortDir = signal<SortDir>(null);
  readonly sortedVarRanking = computed<VarRankingItem[]>(() => {
    const key = this.varSortKey();
    const dir = this.varSortDir();
    const rows = this.varRanking();
    if (!key || !dir) return rows;
    const factor = dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => this.compareVarRow(a, b, key) * factor);
  });

  // ── Pool autocomplete ─────────────────────────────────────────────────
  readonly poolSuggestions = signal<AuctionPlayerSummary[]>([]);
  readonly poolOpen = signal(false);
  private readonly poolQuery$ = new Subject<string>();

  // ── Live form state (plain properties) ───────────────────────────────
  lookupQuery = ''; // display text in lookup input
  lookupId = ''; // resolved playerId
  recordPlayerId = '';
  recordPlayerName = ''; // display text in record input
  recordWinnerId = '';
  recordPrice = 1;
  /** MANTRA: explicit slot; null = backend auto-pick. */
  recordAssignedSlot: string | null = null;
  recordEligibleSlots: string[] = [];
  /** Last selected pool player (for eligible roles / slot UI). */
  recordSelectedPlayer: AuctionPlayerSummary | null = null;

  // ── Initial budgets map (for budget-bar computation) ──────────────────
  private readonly initialBudgets = new Map<string, number>();

  readonly reversedAssignments = computed(() => [...(this.summary()?.assignments ?? [])].reverse());

  /**
   * Cicla lo stato di ordinamento sulla colonna indicata.
   * Sequenza: nessun sort → ascendente → discendente → nessun sort.
   */
  public cycleVarSort(key: VarSortKey): void {
    if (this.varSortKey() !== key) {
      this.varSortKey.set(key);
      this.varSortDir.set('asc');
      return;
    }
    const cur = this.varSortDir();
    if (cur === 'asc') {
      this.varSortDir.set('desc');
    } else if (cur === 'desc') {
      this.varSortDir.set(null);
      this.varSortKey.set(null);
    } else {
      this.varSortDir.set('asc');
    }
  }

  /** Indicatore visuale dell'ordinamento corrente sulla colonna. */
  public varSortIndicator(key: VarSortKey): string {
    if (this.varSortKey() !== key || this.varSortDir() === null) return '↕';
    return this.varSortDir() === 'asc' ? '▲' : '▼';
  }

  /** Valore `aria-sort` per la colonna, conforme ARIA 1.2. */
  public varAriaSort(key: VarSortKey): 'ascending' | 'descending' | 'none' {
    if (this.varSortKey() !== key || this.varSortDir() === null) return 'none';
    return this.varSortDir() === 'asc' ? 'ascending' : 'descending';
  }

  /** Confronto per la colonna di ordinamento; valori null/undefined in fondo. */
  private compareVarRow(a: VarRankingItem, b: VarRankingItem, key: VarSortKey): number {
    const av = this.varSortValue(a, key);
    const bv = this.varSortValue(b, key);
    if (av === bv) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return av < bv ? -1 : 1;
  }

  private varSortValue(item: VarRankingItem, key: VarSortKey): number | string | null | undefined {
    switch (key) {
      case 'name':
        return item.name.toLowerCase();
      case 'role':
        return item.role;
      case 'esv':
        return item.esv;
      case 'expectedPrice':
        return item.expectedPrice;
      case 'seasonValue':
        return item.seasonValue;
      case 'startProbability':
        return item.startProbability;
      case 'buySignal':
        return item.buySignal ? 1 : 0;
    }
  }

  constructor() {
    this.quotationService.getSeasons().subscribe({
      next: (s) => {
        const sorted = [...s].sort((a, b) => b - a);
        this.seasons.set(sorted);
        if (sorted.length) this.seasonStart = sorted[0];
        this.seasonsLoading.set(false);
      },
      error: () => {
        this.seasons.set([...SEASON_FALLBACK_LIST]);
        this.seasonsLoading.set(false);
      },
    });

    // Pool autocomplete: debounce query → call pool endpoint
    this.poolQuery$
      .pipe(
        debounceTime(300),
        distinctUntilChanged(),
        switchMap((q) => {
          const sid = this.sessionId();
          if (!sid) return [];
          return this.auctionService.pool(sid, q);
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (items) => {
          this.poolSuggestions.set(items);
          this.poolOpen.set(items.length > 0);
        },
        error: () => {
          this.poolSuggestions.set([]);
          this.poolOpen.set(false);
        },
      });
  }

  // ── Setup helpers ─────────────────────────────────────────────────────

  /**
   * Handles preset select changes. Applying a preset never clears
   * operator-owned inputs (seasonStart, participants list).
   */
  onPresetChange(presetId: string): void {
    this.selectedPresetId = presetId ?? AUCTION_PRESET_NONE;
    const preset = findAuctionPreset(presetId);
    if (preset) {
      this.applyPreset(preset);
    }
  }

  /**
   * Patches setup form fields from a preset strategy config.
   * League logistics (numParticipants, budgetInitial, referenceBudget,
   * roleQuotas, ruleset) are intentionally left untouched — they belong
   * exclusively to the setup form. Does not start the session and does not
   * mutate participants identities.
   */
  applyPreset(preset: AuctionPreset): void {
    const cfg = preset.config;
    if (cfg.valuationMode === 'PER_MATCH_RATING' || cfg.valuationMode === 'SEASON_VALUE') {
      this.valuationMode = cfg.valuationMode;
    }
    if (cfg.replacementMethod === 'percentile' || cfg.replacementMethod === 'roster_depth') {
      this.replacementMethod = cfg.replacementMethod;
    }
    if (cfg.minStartProbability === null) {
      this.minStartProbability = null;
    } else if (typeof cfg.minStartProbability === 'number') {
      this.minStartProbability = cfg.minStartProbability;
    }

    this.useInflationBaseline = !!cfg.useInflationBaseline;
    const infl = cfg.inflationConfig;
    if (infl) {
      if (infl.inflationPercentileThreshold != null) {
        this.inflationPercentileThreshold = infl.inflationPercentileThreshold;
      }
      if (infl.maxInflationMultiplier != null) {
        this.maxInflationMultiplier = infl.maxInflationMultiplier;
      }
      if (infl.baseInflationRate != null) {
        this.baseInflationRate = infl.baseInflationRate;
      }
      if (infl.baselineParticipants != null) {
        this.baselineParticipants = infl.baselineParticipants;
      }
      if (infl.teamStrengthMultiplier != null) {
        this.teamStrengthMultiplier = infl.teamStrengthMultiplier;
      }
    }

    const md = cfg.marketDriftConfig;
    if (md) {
      if (md.alpha != null) this.alpha = md.alpha;
      if (md.spilloverAdjacentTier != null) this.spilloverAdj = md.spilloverAdjacentTier;
      if (md.spilloverCrossRole != null) this.spilloverCross = md.spilloverCrossRole;
      if (md.minIndex != null) this.minIndex = md.minIndex;
      if (md.maxIndex != null) this.maxIndex = md.maxIndex;
      if (md.tierThresholds?.length === 2) {
        this.tierLow = md.tierThresholds[0];
        this.tierTop = md.tierThresholds[1];
      }
    }

    const alt = cfg.alternativesConfig;
    if (alt?.lowCostPercentile != null) {
      this.lowCostPercentile = alt.lowCostPercentile;
    }

    if (typeof cfg.hybridBlend === 'number') {
      this.hybridBlend = cfg.hybridBlend;
    }

    // Advanced panel is useful when a non-default preset is applied.
    this.showAdvanced = true;
  }

  resizeParticipants(): void {
    this.participants.set(
      makeParticipants(this.numParticipants, this.defaultBudget, this.participants()),
    );
  }

  updateName(i: number, name: string): void {
    this.participants.update((arr) => {
      const next = [...arr];
      next[i] = { ...next[i], displayName: name };
      return next;
    });
  }

  updateBudget(i: number, budget: number): void {
    this.participants.update((arr) => {
      const next = [...arr];
      next[i] = { ...next[i], budgetInitial: budget };
      return next;
    });
  }

  /** Propaga "Budget each" a tutti i participants. */
  applyDefaultBudget(): void {
    this.participants.update((arr) =>
      arr.map((p) => ({ ...p, budgetInitial: this.defaultBudget })),
    );
  }

  // ── Session init ──────────────────────────────────────────────────────

  /** Snapshot of the setup form as AuctionConfig (used by simulation satellite). */
  get setupAuctionConfig(): AuctionConfig {
    return {
      numParticipants: this.numParticipants,
      roleQuotas: this.roleQuotas,
      ruleset: this.ruleset,
      hybridBlend: this.hybridBlend,
      marketDriftConfig: {
        alpha: this.alpha,
        spilloverAdjacentTier: this.spilloverAdj,
        spilloverCrossRole: this.spilloverCross,
        minIndex: this.minIndex,
        maxIndex: this.maxIndex,
        tierThresholds: [this.tierLow, this.tierTop],
      },
      alternativesConfig: { lowCostPercentile: this.lowCostPercentile },
      useInflationBaseline: this.useInflationBaseline,
      ...(this.useInflationBaseline
        ? {
            inflationConfig: {
              inflationPercentileThreshold: this.inflationPercentileThreshold,
              maxInflationMultiplier: this.maxInflationMultiplier,
              baseInflationRate: this.baseInflationRate,
              baselineParticipants: this.baselineParticipants,
              teamStrengthMultiplier: this.teamStrengthMultiplier,
            },
          }
        : {}),
      minStartProbability: this.minStartProbability,
      replacementMethod: this.replacementMethod,
      referenceBudget: this.referenceBudget,
      budgetInitial: this.budgetInitial,
      valuationMode: this.valuationMode,
    };
  }

  startAuction(): void {
    this.starting.set(true);
    this.initError.set(null);
    this._cacheInitialBudgets(this.participants());

    this.auctionService
      .init({
        seasonStart: this.seasonStart,
        participants: this.participants(),
        config: {
          numParticipants: this.numParticipants,
          roleQuotas: this.roleQuotas,
          ruleset: this.ruleset,
          hybridBlend: this.hybridBlend,
          marketDriftConfig: {
            alpha: this.alpha,
            spilloverAdjacentTier: this.spilloverAdj,
            spilloverCrossRole: this.spilloverCross,
            minIndex: this.minIndex,
            maxIndex: this.maxIndex,
            tierThresholds: [this.tierLow, this.tierTop],
          },
          alternativesConfig: { lowCostPercentile: this.lowCostPercentile },
          useInflationBaseline: this.useInflationBaseline,
          // Only attach the inflationConfig object when the baseline is enabled
          // and the user hasn't reset the values. Mirrors the backend's
          // Optional[InflationConfigSchema] contract: omitted ⇒ server defaults.
          ...(this.useInflationBaseline
            ? {
                inflationConfig: {
                  inflationPercentileThreshold: this.inflationPercentileThreshold,
                  maxInflationMultiplier: this.maxInflationMultiplier,
                  baseInflationRate: this.baseInflationRate,
                  baselineParticipants: this.baselineParticipants,
                  teamStrengthMultiplier: this.teamStrengthMultiplier,
                },
              }
            : {}),
          // Pool pre-filter and replacement level are always sent so the
          // server-side VarEngine applies them regardless of the inflation
          // toggle (they affect the ranking, not the price model).
          minStartProbability: this.minStartProbability,
          replacementMethod: this.replacementMethod,
          referenceBudget: this.referenceBudget,
          budgetInitial: this.budgetInitial,
          valuationMode: this.valuationMode,
        },
      })
      .subscribe({
        next: (res) => {
          this.sessionId.set(res.sessionId);
          this.nExcludedNoProjection.set(res.nExcludedNoProjection ?? 0);
          this.starting.set(false);
          this.refreshSummary();
        },
        error: (err) => {
          this.initError.set(err.error?.detail ?? 'Failed to start session');
          this.starting.set(false);
        },
      });
  }

  onResumeFile(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const payload = JSON.parse(e.target!.result as string);
        this.starting.set(true);
        this.initError.set(null);
        this.auctionService.deserialize({ payload }).subscribe({
          next: (res) => {
            this.sessionId.set(res.sessionId);
            this.starting.set(false);
            this.refreshSummary();
          },
          error: (err) => {
            this.initError.set(err.error?.detail ?? 'Failed to resume session');
            this.starting.set(false);
          },
        });
      } catch {
        this.initError.set('Invalid save file — must be JSON');
      }
    };
    reader.readAsText(file);
    // Reset input so same file can be re-selected
    (event.target as HTMLInputElement).value = '';
  }

  // ── Live actions ──────────────────────────────────────────────────────

  onPoolQueryChange(q: string): void {
    this.poolQuery$.next(q);
    if (!q.trim()) {
      this.poolSuggestions.set([]);
      this.poolOpen.set(false);
    }
  }

  selectPoolPlayer(p: AuctionPlayerSummary): void {
    this.lookupId = p.playerId;
    this.lookupQuery = p.name;
    // pre-fill record card too
    this.recordPlayerId = p.playerId;
    this.recordPlayerName = `${p.name} (${this.roleLabel(p)} · ${p.realTeam})`;
    this.recordSelectedPlayer = p;
    this.recordEligibleSlots =
      p.eligibleRoles && p.eligibleRoles.length > 0
        ? [...p.eligibleRoles]
        : this.ruleset === 'MANTRA'
          ? [p.role]
          : [];
    this.recordAssignedSlot =
      this.recordEligibleSlots.length === 1 ? this.recordEligibleSlots[0] : null;
    this.poolOpen.set(false);
    this.poolSuggestions.set([]);
    this.lookupPlayer(p.playerId);
  }

  lookupPlayer(playerId = this.lookupId): void {
    const sid = this.sessionId();
    if (!sid || !playerId) return;
    this.lookupLoading.set(true);
    this.lookupError.set(null);
    this.projection.set(null);
    this.altResult.set(null);

    forkJoin({
      proj: this.auctionService.projection(sid, playerId),
      alt: this.auctionService.alternatives(sid, playerId, {
        participantId: this.recordWinnerId || null,
        strategyName: this.altStrategyName,
      }),
    }).subscribe({
      next: ({ proj, alt }) => {
        this.projection.set(proj);
        this.altResult.set(alt);
        this.lookupLoading.set(false);
      },
      error: (err) => {
        this.lookupError.set(err.error?.detail ?? 'Player not found');
        this.lookupLoading.set(false);
      },
    });
  }

  /** Refresh alternatives when the winner changes (max bid depends on residual budget). */
  onWinnerChange(participantId: string): void {
    this.recordWinnerId = participantId;
    if (this.lookupId || this.recordPlayerId) {
      this.lookupPlayer(this.lookupId || this.recordPlayerId);
    }
  }

  /** Refresh alternatives when strategy cap changes. */
  onStrategyChange(name: string | null): void {
    this.altStrategyName = name;
    if (this.lookupId || this.recordPlayerId) {
      this.lookupPlayer(this.lookupId || this.recordPlayerId);
    }
  }

  submitRecord(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.recordLoading.set(true);
    this.recordError.set(null);
    this.recordRejectionCode.set(null);

    this.auctionService
      .record(sid, {
        playerId: this.recordPlayerId,
        winnerParticipantId: this.recordWinnerId,
        finalPrice: this.recordPrice,
        assignedSlot: this.recordAssignedSlot,
      })
      .subscribe({
        next: (res) => {
          if (!res.success) {
            this.recordError.set(res.rejectionReason ?? 'Assignment rejected');
            this.recordRejectionCode.set(res.rejectionCode ?? null);
          } else {
            this.recordPlayerId = '';
            this.recordPlayerName = '';
            this.recordWinnerId = '';
            this.recordPrice = 1;
            this.recordAssignedSlot = null;
            this.recordEligibleSlots = [];
            this.recordSelectedPlayer = null;
            this.recordError.set(null);
            this.projection.set(null);
            this.altResult.set(null);
            this.lookupId = '';
            this.lookupQuery = '';
            this.refreshSummary();
          }
          this.recordLoading.set(false);
        },
        error: (err) => {
          this.recordError.set(err.error?.detail ?? 'Server error');
          this.recordLoading.set(false);
        },
      });
  }

  undoLast(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.undoLoading.set(true);
    this.auctionService.undo(sid).subscribe({
      next: () => {
        this.undoLoading.set(false);
        this.refreshSummary();
      },
      error: () => this.undoLoading.set(false),
    });
  }

  saveToFile(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.auctionService.serialize(sid).subscribe({
      next: (res) => {
        const blob = new Blob([JSON.stringify(res.payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `auction_${sid.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      },
    });
  }

  endSession(): void {
    if (!confirm('End this auction session? The session will be deleted.')) return;
    const sid = this.sessionId();
    if (!sid) return;
    this.auctionService.discard(sid).subscribe({
      next: () => this._resetLiveState(),
      error: () => this._resetLiveState(),
    });
  }

  refreshSummary(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.summaryLoading.set(true);
    this.auctionService.summary(sid).subscribe({
      next: (s) => {
        this.summary.set(s);
        this.summaryLoading.set(false);
        // Populate initialBudgets from setup participants if not already set
        if (this.initialBudgets.size === 0) {
          s.participants.forEach((p) => {
            if (!this.initialBudgets.has(p.participantId)) {
              this.initialBudgets.set(p.participantId, p.budgetResidual);
            }
          });
        }
        this.refreshVarRanking();
      },
      error: () => this.summaryLoading.set(false),
    });
  }

  refreshVarRanking(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.varLoading.set(true);
    this.auctionService.varRanking(sid).subscribe({
      next: (res) => {
        this.varRanking.set(res.items);
        this.varLoading.set(false);
      },
      error: () => this.varLoading.set(false),
    });
  }

  // ── Template helpers ──────────────────────────────────────────────────


  openVarPlayer(v: VarRankingItem): void {
    this.selectedPlayer.set({
      playerId: v.playerId,
      name: v.name,
      role: v.role,
      projectedScore: v.projectedScore,
      varScore: v.varScore,
      expectedPrice: v.expectedPrice,
      esv: v.esv,
      calibrated: v.calibrated,
      buySignal: v.buySignal,
      seasonValue: v.seasonValue,
      startProbability: v.startProbability,
      sampleCohort: v.sampleCohort,
      reliabilityWeight: v.reliabilityWeight,
    });
  }

  openAssignmentPlayer(a: AssignmentRecord): void {
    this.selectedPlayer.set({
      playerId: a.player.playerId,
      name: a.player.name,
      role: a.player.role,
      realTeam: a.player.realTeam,
      cost: a.player.cost,
      projectedScore: a.player.projectedScore,
      finalPrice: a.finalPrice,
      tier: a.tier,
      sampleCohort: a.player.sampleCohort,
      reliabilityWeight: a.player.reliabilityWeight,
    });
  }

  roleColor(role: string): string {
    return ROLE_COLOR[role] ?? 'var(--color-text-secondary)';
  }
  tierColor(tier: AuctionTier): string {
    return TIER_COLOR[tier];
  }

  budgetPercent(p: AuctionParticipantState): number {
    const initial = this.initialBudgets.get(p.participantId) ?? p.budgetResidual;
    if (initial === 0) return 0;
    return Math.max(0, Math.min(100, (p.budgetResidual / initial) * 100));
  }

  budgetColor(p: AuctionParticipantState): string {
    const pct = this.budgetPercent(p) / 100;
    if (pct > 0.4) return 'var(--color-text-primary)';
    if (pct > 0.2) return '#F59E0B';
    return '#EF4444';
  }

  winnerName(participantId: string): string {
    return (
      this.summary()?.participants.find((p) => p.participantId === participantId)?.displayName ??
      participantId
    );
  }

  // ── Private ───────────────────────────────────────────────────────────

  private _cacheInitialBudgets(participants: AuctionParticipantSetup[]): void {
    this.initialBudgets.clear();
    participants.forEach((p) => this.initialBudgets.set(p.participantId, p.budgetInitial));
  }

  private _resetLiveState(): void {
    this.sessionId.set(null);
    this.nExcludedNoProjection.set(0);
    this.summary.set(null);
    this.projection.set(null);
    this.altResult.set(null);
    this.poolSuggestions.set([]);
    this.poolOpen.set(false);
    this.lookupId = '';
    this.lookupQuery = '';
    this.recordPlayerId = '';
    this.recordPlayerName = '';
    this.recordWinnerId = '';
    this.recordPrice = 1;
    this.initialBudgets.clear();
  }
}