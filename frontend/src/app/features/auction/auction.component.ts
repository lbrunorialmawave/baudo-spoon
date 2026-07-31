import {
  Component, computed, inject, signal, DestroyRef,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { forkJoin, Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { AuctionService } from '../../core/services/auction.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  AUCTION_ROLES,
  AuctionParticipantSetup,
  AuctionParticipantState,
  AuctionPlayerSummary,
  AuctionRole,
  AuctionSummary,
  AuctionTier,
  ProjectionResponse,
  AlternativesResponse,
  ValuationMode,
  VarRankingItem,
} from '../../core/models/auction.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { FieldLegendComponent, FieldLegendExample } from '../../shared/components/field-legend/field-legend.component';
import { OPTIMIZER_LEGENDS } from '../optimizer/optimizer.component';

const ROLE_COLOR: Record<string, string> = {
  P: 'var(--color-role-gk)',
  D: 'var(--color-role-def)',
  C: 'var(--color-role-mid)',
  A: 'var(--color-role-fwd)',
};

const TIER_COLOR: Record<AuctionTier, string> = {
  LOW: 'var(--color-text-secondary)',
  MID: 'var(--color-accent)',
  TOP: '#F59E0B',
};

/** Colonne ordinabili della tabella "Migliori affari". */
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
const SETUP_LEGENDS: Readonly<Record<string, { description: string; examples: readonly FieldLegendExample[] }>> = {
  seasonStart: {
    description: 'Anno di inizio della stagione di Serie A da usare come riferimento per le quotazioni storiche, le statistiche per-match e i listini impiegati dal risolutore. Determina anche la finestra temporale delle partite considerate per il calcolo dei tier.',
    examples: [
      { label: '2025', value: 'stagione 2025/26 (default più recente)' },
      { label: '2024', value: 'stagione 2024/25 (storica)' },
      { label: '2023', value: 'stagione 2023/24 (analisi trend)' },
    ],
  },
  numParticipants: {
    description: 'Numero di squadre che partecipano all\'asta, cioè il numero di giocatori-manager che si contenderanno i calciatori. Influisce sul budget totale in gioco e sulla pressione dei prezzi: più partecipanti significano maggiore competizione.',
    examples: [
      { label: '4', value: 'lega piccola tra amici' },
      { label: '8', value: 'lega standard (default consigliato)' },
      { label: '10–12', value: 'lega grande, alta competizione' },
    ],
  },
  defaultBudget: {
    description: 'Crediti iniziali (Fantamiliardi, abbreviato "cr.") assegnati a ogni partecipante per costruire la rosa. Se modifichi questo valore, viene applicato a tutti i partecipanti già presenti in lista tramite il pulsante "Applica a tutti".',
    examples: [
      { label: '300 cr.', value: 'lega piccola' },
      { label: '500 cr.', value: 'default classico' },
      { label: '1000 cr.', value: 'lega premium con top player costosi' },
    ],
  },
  roleQuotas: {
    description: 'Numero massimo di giocatori per ruolo che ogni squadra può avere in rosa (Portieri, Difensori, Centrocampisti, Attaccanti). Le quote devono essere coerenti con il regolamento della lega.',
    examples: [
      { label: 'P=3 D=8 C=8 A=6', value: 'rosa 25, modulo 3-5-2 con slot panchina' },
      { label: 'P=3 D=8 C=8 A=6', value: 'default Mantra-style con 3 cambi ruolo' },
    ],
  },
  useInflationBaseline: {
    description: 'Se attivo, il sistema applica un modello di inflazione che stima quanto i prezzi saliranno rispetto al listino base, basandosi sul numero di partecipanti. Disattivalo solo se vuoi prezzi "piatti" senza rialzo previsto.',
    examples: [
      { label: 'ON', value: 'prezzi EWMA gonfiati dalla competizione' },
      { label: 'OFF', value: 'prezzi allineati al listino base' },
    ],
  },
  referenceBudget: {
    description: 'Budget di riferimento (in crediti) usato dal modello di inflazione come "lega neutra" per calibrare i moltiplicatori. Tipicamente 300 cr. per leghe standard; alzalo se la tua lega usa budget più alti.',
    examples: [
      { label: '300 cr.', value: 'riferimento per leghe 300 cr.' },
      { label: '500 cr.', value: 'riferimento per leghe 500 cr.' },
    ],
  },
  budgetInitial: {
    description: 'Budget di partenza effettivo di questa sessione d\'asta, in crediti. Impostalo sul valore reale della tua lega: è il numero che il tracker userà per calcolare il residuo di ogni partecipante.',
    examples: [
      { label: '300 cr.', value: 'lega classica FantaSanremo/Mantra' },
      { label: '500 cr.', value: 'lega premium' },
    ],
  },
  valuationMode: {
    description: 'Modalità con cui viene calcolato il valore atteso di un giocatore durante l\'asta. "Per-Match Rating" usa la media voto ponderata per partita; "Season Value" usa una stima complessiva di fine stagione più adatta a leghe con premi stagionali.',
    examples: [
      { label: 'PER_MATCH_RATING', value: 'lega settimanale, focus sul rendimento singolo' },
      { label: 'SEASON_VALUE', value: 'lega con bonus stagionali' },
    ],
  },
  alpha: {
    description: 'Fattore di smoothing dell\'indice EWMA (Exponentially Weighted Moving Average). Determina quanto peso ha l\'ultima osservazione rispetto alla storia passata: valori alti reagiscono velocemente ai rialzi, valori bassi smussano il mercato.',
    examples: [
      { label: '0.1', value: 'molto stabile, ideale per leghe tranquille' },
      { label: '0.3', value: 'default bilanciato' },
      { label: '0.6', value: 'reattivo, adatto ad aste molto inflazionate' },
    ],
  },
  spilloverAdj: {
    description: 'Coefficiente (0–1) di "spillover" che trasferisce parte della pressione di prezzo dallo stesso ruolo al tier adiacente (es. MID → LOW o MID → TOP). 0 = nessuno spillover; 0.1 = leggera propagazione; 0.3 = forte.',
    examples: [
      { label: '0.05', value: 'spillover leggero (default)' },
      { label: '0.20', value: 'spillover marcato, mercato liquido' },
    ],
  },
  spilloverCross: {
    description: 'Coefficiente (0–1) di spillover cross-ruolo: quanto un rialzo su un ruolo (es. Attaccanti) si propaga a un altro ruolo (es. Centrocampisti). Utile in leghe con regole di conversione ruolo.',
    examples: [
      { label: '0.0', value: 'nessuna propagazione cross-ruolo' },
      { label: '0.05', value: 'default leggera' },
      { label: '0.15', value: 'propagazione forte' },
    ],
  },
  lowCostPercentile: {
    description: 'Percentile (0–1) sotto il quale un giocatore è considerato "low cost" dal sistema di suggerimento alternative. Influenza la proposta "Low Cost" mostrata nel lookup.',
    examples: [
      { label: '0.2', value: 'solo i giocatori più economici (top 20%)' },
      { label: '0.3', value: 'default' },
      { label: '0.5', value: 'fascia media-bassa' },
    ],
  },
  minIndex: {
    description: 'Valore minimo consentito per l\'indice di prezzo EWMA dopo l\'applicazione del modello di inflazione. Impedisce che i prezzi crollino a zero in leghe con poca domanda.',
    examples: [
      { label: '0.5', value: 'default, previene crolli sotto listino' },
      { label: '0.8', value: 'floor più alto, mercato rialzistico' },
    ],
  },
  maxIndex: {
    description: 'Valore massimo consentito per l\'indice di prezzo EWMA. Impedisce runaway inflation in leghe molto competitive, fissando un tetto oltre il quale il prezzo non può salire.',
    examples: [
      { label: '2.0', value: 'default (×2 rispetto al listino)' },
      { label: '3.0', value: 'lega molto calda, top player esplosivi' },
    ],
  },
  tierLow: {
    description: 'Soglia (0–1) sotto la quale un giocatore è classificato tier LOW. È una soglia relativa: il sistema normalizza gli indici EWMA del ruolo in [0,1] e poi confronta con questo valore.',
    examples: [
      { label: '0.3', value: 'default, 30% dei giocatori nel tier basso' },
      { label: '0.5', value: 'più giocatori finisco nel tier basso' },
    ],
  },
  tierTop: {
    description: 'Soglia (0–1) sopra la quale un giocatore è classificato tier TOP. Insieme a `tierLow` definisce la fascia centrale (tier MID).',
    examples: [
      { label: '0.7', value: 'default, 30% top del ruolo' },
      { label: '0.85', value: 'top ristretto ai migliori assoluti' },
    ],
  },
};

const LIVE_LEGENDS: Readonly<Record<string, { description: string; examples: readonly FieldLegendExample[] }>> = {
  lookupQuery: {
    description: 'Casella di ricerca libera sulla pool di giocatori disponibili per l\'asta. La ricerca è debouncata di 300 ms e mostra un dropdown con i risultati migliori (per nome, squadra e ruolo).',
    examples: [
      { label: '"Lautaro"', value: 'autocomplete su cognome / parte del nome' },
      { label: '"Inter A"', value: 'filtra per team + ruolo' },
      { label: 'Esc', value: 'chiude il dropdown' },
      { label: 'Invio', value: 'esegue il lookup del giocatore attualmente nel campo' },
    ],
  },
  recordPlayer: {
    description: 'Giocatore attualmente selezionato per la registrazione dell\'assegnazione. Viene pre-compilato automaticamente scegliendo un risultato dal Lookup, ma può essere sovrascritto manualmente incollando un ID.',
    examples: [
      { label: 'selezione', value: 'scegli dal Lookup →' },
      { label: 'ID manuale', value: 'incolla un fm-XXXX se conosci l\'identificativo FotMob' },
    ],
  },
  recordWinner: {
    description: 'Squadra partecipante che si aggiudica il giocatore in questo turno d\'asta. La lista deriva dai partecipanti configurati nella sessione; l\'opzione vuota disabilita il pulsante Registra.',
    examples: [
      { label: 'Team 1', value: 'aggiudicatario turno corrente' },
      { label: '— select —', value: 'nessuna selezione (pulsante Registra disabilitato)' },
    ],
  },
  recordPrice: {
    description: 'Prezzo finale di aggiudicazione, in crediti (cr.). Deve essere un intero ≥ 1. Verrà scalato dal budget residuo del vincitore e alimenterà l\'indice EWMA per quel ruolo/tier.',
    examples: [
      { label: '1', value: 'prezzo minimo, svincolo low-cost' },
      { label: '30–50', value: 'tier MID tipico' },
      { label: '100+', value: 'tier TOP, top player' },
    ],
  },
  varRanking: {
    description: 'Tabella "Migliori affari": classifica dei giocatori con ESV (Expected Season Value) positivo, ordinata per rendimento atteso rispetto al prezzo EWMA. La colonna "Segnale" indica se il sistema consiglia l\'acquisto (COMPRA) come affare.',
    examples: [
      { label: 'ESV > 0 + COMPRA', value: 'affare: valore atteso superiore al prezzo' },
      { label: 'ESV ≤ 0 + -', value: 'surriscaldato, prezzo troppo alto rispetto al rendimento previsto' },
    ],
  },
};

function makeParticipants(
  n: number, budget: number, existing: AuctionParticipantSetup[] = [],
): AuctionParticipantSetup[] {
  return Array.from({ length: n }, (_, i) => existing[i] ?? {
    participantId: `team_${i + 1}`,
    displayName: `Team ${i + 1}`,
    budgetInitial: budget,
  });
}

@Component({
  selector: 'app-auction',
  standalone: true,
  imports: [FormsModule, DecimalPipe, SkeletonComponent, ErrorBoundaryComponent, FieldLegendComponent],
  template: `
    @if (sessionId()) {

      <!-- ═══════════════════════ LIVE VIEW ═══════════════════════ -->
      <div class="auction-page">

        <header class="page-header">
          <div>
            <h1 class="page-title">Tracker Asta</h1>
            <p class="page-subtitle">
              Sessione attiva: <code class="session-id">{{ sessionId()!.slice(0, 12) }}…</code>
            </p>
          </div>
          <div class="header-actions">
            <button class="secondary-btn" (click)="saveToFile()" title="Esporta l'intera sessione (assegnazioni, budget, indici EWMA) in un file JSON che potrai ricaricare in seguito.">Salva sessione</button>
            <button class="danger-btn" (click)="endSession()" title="Termina e cancella definitivamente la sessione dal backend. Operazione irreversibile.">Termina sessione</button>
          </div>
        </header>

        <!-- Price index strip -->
        @if (summary(); as s) {
          <div class="price-strip" title="Indice di prezzo EWMA corrente per ogni combinazione ruolo × tier. Più alto = più caro del listino base.">
            @for (role of allRoles; track role) {
              <div class="price-role-group">
                <span class="price-role-label" [style.color]="roleColor(role)">{{ role }}</span>
                @for (tier of allTiers; track tier) {
                  @if (s.priceIndex[role]?.[tier] !== undefined) {
                    <span class="price-chip" [style.border-color]="tierColor(tier)"
                          [style.color]="tierColor(tier)">
                      {{ tier.charAt(0) }}&thinsp;{{ s.priceIndex[role]![tier]! | number:'1.2-2' }}
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
            <p class="panel-heading">Classifica e budget residuo</p>

            @if (summaryLoading() && !summary()) {
              @for (_ of [1,2,3,4,5,6,7,8]; track $index) {
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
                    <div class="budget-bar-fill"
                         [style.width]="budgetPercent(p) + '%'"
                         [style.background]="budgetColor(p)"></div>
                  </div>
                  <div class="role-chips">
                    @for (role of allRoles; track role) {
                      @if (p.roleBreakdown[role]) {
                        <span class="role-chip"
                              [style.color]="roleColor(role)"
                              [style.border-color]="roleColor(role)">
                          {{ role }}&thinsp;{{ p.roleBreakdown[role] }}
                        </span>
                      }
                    }
                    @if (p.squad.length === 0) {
                      <span class="empty-squad" title="Nessun giocatore ancora acquistato">—</span>
                    }
                  </div>
                </div>
              }
            }
          </aside>

          <!-- ── Right: Main area ──────────────────────── -->
          <main class="auction-main">

            <div class="action-row">

              <!-- Lookup card -->
              <div class="card">
                <p class="card-section-label">Ricerca giocatore (Lookup)</p>
                <div class="pool-autocomplete">
                  <div class="lookup-row">
                    <input id="lookupQuery" class="field-input" placeholder="Cerca giocatore per nome, squadra o ruolo…"
                           [ngModel]="lookupQuery"
                           (ngModelChange)="lookupQuery = $event; onPoolQueryChange($event)"
                           (keydown.escape)="poolOpen.set(false)"
                           (keydown.enter)="lookupPlayer()"
                           [attr.aria-describedby]="'legend-lookupQuery'"
                           autocomplete="off" />
                    @if (lookupLoading()) {
                      <span class="spinner-sm" style="flex-shrink:0;color:var(--color-accent)" aria-label="Caricamento suggerimenti"></span>
                    }
                  </div>
                  <app-field-legend
                    fieldId="legend-lookupQuery"
                    [description]="LIVE_LEGENDS['lookupQuery'].description"
                    [examples]="LIVE_LEGENDS['lookupQuery'].examples" />
                  @if (poolOpen() && poolSuggestions().length) {
                    <ul class="pool-dropdown" role="listbox">
                      @for (p of poolSuggestions(); track p.playerId) {
                        <li class="pool-option" role="option"
                            (mousedown)="selectPoolPlayer(p)">
                          <span class="pool-name">{{ p.name }}</span>
                          <span class="pool-meta">
                            <span class="role-badge"
                                  [style.color]="roleColor(p.role)"
                                  [style.border-color]="roleColor(p.role)">{{ p.role }}</span>
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
                  <div class="projection-row" title="Prezzo atteso EWMA per il giocatore: è la stima di quanto dovrebbe essere aggiudicato in questo momento dell'asta.">
                    <span class="proj-label">Prezzo atteso</span>
                    <span class="proj-price">{{ proj.expectedPrice | number:'1.0-0' }} cr.</span>
                    <span class="tier-badge" [style.color]="tierColor(proj.tier)"
                          [style.border-color]="tierColor(proj.tier)">{{ proj.tier }}</span>
                  </div>
                }

                @if (altResult(); as alt) {
                  <div class="alternatives-grid">
                    @if (alt.lowCostAlternative; as lc) {
                      <div class="alt-card">
                        <p class="alt-label">Alternativa economica</p>
                        <p class="alt-name">{{ lc.name }}</p>
                        <p class="alt-meta">{{ lc.realTeam }} · {{ lc.role }} · {{ lc.cost }} cr.</p>
                      </div>
                    }
                    @if (alt.closestAlternative; as cl) {
                      <div class="alt-card">
                        <p class="alt-label">Alternativa più simile</p>
                        <p class="alt-name">{{ cl.name }}</p>
                        <p class="alt-meta">{{ cl.realTeam }} · {{ cl.role }} · {{ cl.cost }} cr.</p>
                      </div>
                    }
                    @if (!alt.lowCostAlternative && !alt.closestAlternative && alt.reasonIfNone) {
                      <p class="alt-none">{{ alt.reasonIfNone }}</p>
                    }
                  </div>
                }
              </div>

              <!-- Record card -->
              <div class="card">
                <p class="card-section-label">Registra assegnazione di turno</p>

                <div class="field-group">
                  <label class="field-label" for="recordPlayer">Giocatore</label>
                  <input id="recordPlayer" class="field-input" [ngModel]="recordPlayerName || recordPlayerId"
                         readonly placeholder="seleziona dal Lookup →"
                         [style.color]="recordPlayerId ? 'var(--color-text-primary)' : 'var(--color-text-secondary)'"
                         [attr.aria-describedby]="'legend-recordPlayer'" />
                  <app-field-legend
                    fieldId="legend-recordPlayer"
                    [description]="LIVE_LEGENDS['recordPlayer'].description"
                    [examples]="LIVE_LEGENDS['recordPlayer'].examples" />
                </div>

                <div class="field-group">
                  <label class="field-label" for="recordWinner">Squadra vincitrice (aggiudicatario)</label>
                  <select id="recordWinner" class="field-input" [(ngModel)]="recordWinnerId"
                          [attr.aria-describedby]="'legend-recordWinner'">
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
                    [examples]="LIVE_LEGENDS['recordWinner'].examples" />
                </div>

                <div class="field-group">
                  <label class="field-label" for="recordPrice">Prezzo finale di aggiudicazione <span class="field-hint">cr.</span></label>
                  <input id="recordPrice" class="field-input" type="number" min="1"
                         [(ngModel)]="recordPrice"
                         [attr.aria-describedby]="'legend-recordPrice'" />
                  <app-field-legend
                    fieldId="legend-recordPrice"
                    [description]="LIVE_LEGENDS['recordPrice'].description"
                    [examples]="LIVE_LEGENDS['recordPrice'].examples" />
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
                  <button class="run-btn"
                          (click)="submitRecord()"
                          [disabled]="recordLoading() || !recordPlayerId || !recordWinnerId || recordPrice < 1"
                          title="Conferma l'assegnazione: scalo il prezzo dal budget del vincitore, aggiorno l'indice EWMA del ruolo/tier e aggiungo l'operazione allo storico.">
                    @if (recordLoading()) {
                      <span class="spinner"></span> Registrazione…
                    } @else {
                      Registra assegnazione
                    }
                  </button>
                  <button class="secondary-btn" (click)="undoLast()" [disabled]="undoLoading()"
                          title="Annulla l'ultima assegnazione registrata: ripristina il budget del vincitore e ripristina l'indice EWMA al valore precedente.">
                    @if (undoLoading()) { <span class="spinner-sm"></span> } @else { Annulla ultima }
                  </button>
                </div>
              </div>

            </div>

            <!-- Assignment history -->
            @if (reversedAssignments().length) {
              <div class="card history-card">
                <p class="card-section-label">Storico assegnazioni ({{ reversedAssignments().length }})</p>
                <div class="history-table-wrap">
                  <table class="squad-table">
                    <thead>
                      <tr>
                        <th title="Numero progressivo dell'asta nella sessione">#</th>
                        <th>Giocatore</th>
                        <th>Aggiudicatario</th>
                        <th title="Ruolo del giocatore (P/D/C/A)">Ruolo</th>
                        <th class="num" title="Prezzo finale di aggiudicazione in crediti">Prezzo</th>
                        <th title="Tier di quotazione al momento dell'asta">Tier</th>
                        <th class="num" title="Variazione dell'indice EWMA del ruolo/tier dopo l'assegnazione">Δ Indice</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (a of reversedAssignments(); track a.sequenceNumber) {
                        <tr>
                          <td class="seq">{{ a.sequenceNumber }}</td>
                          <td>
                            <p class="player-name">{{ a.player.name }}</p>
                            <p class="team-name">{{ a.player.realTeam }}</p>
                          </td>
                          <td>{{ winnerName(a.winnerParticipantId) }}</td>
                          <td>
                            <span class="role-badge"
                                  [style.color]="roleColor(a.role)"
                                  [style.border-color]="roleColor(a.role)">{{ a.role }}</span>
                          </td>
                          <td class="num accent">{{ a.finalPrice }}</td>
                          <td>
                            <span class="tier-badge"
                                  [style.color]="tierColor(a.tier)"
                                  [style.border-color]="tierColor(a.tier)">{{ a.tier }}</span>
                          </td>
                          <td class="num faded">
                            {{ (a.priceIndexAfter - a.priceIndexBefore) | number:'+1.3-3' }}
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            } @else if (summary()) {
              <div class="card empty-history">
                <p>Nessuna assegnazione registrata. Cerca un giocatore nel Lookup, poi registra la sua vendita.</p>
              </div>
            }

            <!-- VAR/ESV ranking -->
            @if (varRanking().length || varLoading()) {
              <div class="card">
                <p class="card-section-label">
                  Migliori affari disponibili (ESV vs prezzo EWMA)
                  @if (varLoading()) { <span class="spinner-sm" style="margin-left:6px"></span> }
                </p>
                <app-field-legend
                  fieldId="legend-varRanking"
                  [description]="LIVE_LEGENDS['varRanking'].description"
                  [examples]="LIVE_LEGENDS['varRanking'].examples" />
                @if (varRanking().length) {
                  <div class="history-table-wrap">
                    <table class="squad-table">
                      <thead>
                        <tr>
                          <th class="sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('name')"
                              (click)="cycleVarSort('name')"
                              (keydown.enter)="cycleVarSort('name')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('name')"
                              title="Ordina per nome del giocatore">
                            <span>Giocatore</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('name') }}</span>
                          </th>
                          <th class="sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('role')"
                              (click)="cycleVarSort('role')"
                              (keydown.enter)="cycleVarSort('role')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('role')"
                              title="Ordina per ruolo (P/D/C/A)">
                            <span>Ruolo</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('role') }}</span>
                          </th>
                          <th class="num sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('esv')"
                              (click)="cycleVarSort('esv')"
                              (keydown.enter)="cycleVarSort('esv')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('esv')"
                              title="Ordina per Expected Season Value">
                            <span>ESV</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('esv') }}</span>
                          </th>
                          <th class="num sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('expectedPrice')"
                              (click)="cycleVarSort('expectedPrice')"
                              (keydown.enter)="cycleVarSort('expectedPrice')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('expectedPrice')"
                              title="Ordina per prezzo atteso EWMA">
                            <span>Prezzo atteso</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('expectedPrice') }}</span>
                          </th>
                          <th class="num sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('seasonValue')"
                              (click)="cycleVarSort('seasonValue')"
                              (keydown.enter)="cycleVarSort('seasonValue')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('seasonValue')"
                              title="Ordina per valore di stagione">
                            <span>Val. stagione</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('seasonValue') }}</span>
                          </th>
                          <th class="num sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('startProbability')"
                              (click)="cycleVarSort('startProbability')"
                              (keydown.enter)="cycleVarSort('startProbability')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('startProbability')"
                              title="Ordina per probabilità di titolarità">
                            <span>% Titolarità</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('startProbability') }}</span>
                          </th>
                          <th class="sortable" tabindex="0" role="columnheader"
                              [attr.aria-sort]="varAriaSort('buySignal')"
                              (click)="cycleVarSort('buySignal')"
                              (keydown.enter)="cycleVarSort('buySignal')"
                              (keydown.space)="$event.preventDefault(); cycleVarSort('buySignal')"
                              title="Ordina per segnale di acquisto (COMPRA prima)">
                            <span>Segnale</span>
                            <span class="sort-indicator" aria-hidden="true">{{ varSortIndicator('buySignal') }}</span>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        @for (v of sortedVarRanking(); track v.playerId) {
                          <tr>
                            <td>{{ v.name }}</td>
                            <td>
                              <span class="role-badge"
                                    [style.color]="roleColor(v.role)"
                                    [style.border-color]="roleColor(v.role)">{{ v.role }}</span>
                            </td>
                            <td class="num" [style.color]="v.esv > 0 ? 'var(--color-success, #22C55E)' : 'var(--color-text-secondary)'">
                              {{ v.esv | number:'1.1-1' }}
                            </td>
                            <td class="num faded">{{ v.expectedPrice | number:'1.0-0' }}</td>
                            <td class="num faded">{{ v.seasonValue != null ? (v.seasonValue | number:'1.1-1') : '—' }}</td>
                            <td class="num faded">{{ v.startProbability != null ? (v.startProbability * 100 | number:'1.0-0') + '%' : '—' }}</td>
                            <td>
                              @if (v.buySignal) {
                                <span class="esv-badge esv-buy" title="Affare: ESV positivo">COMPRA</span>
                              } @else {
                                <span class="esv-badge esv-hold" title="EVITARE: ESV negativo o nullo">-</span>
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
            <p class="page-subtitle">Indice di mercato EWMA con modello di inflazione basato sul numero di partecipanti</p>
          </div>
        </header>

        <div class="setup-body">

          <!-- Config panel -->
          <aside class="config-panel card">

            <p class="section-divider">Sessione</p>

            <div class="field-group">
              <label class="field-label" for="seasonStart">Stagione di riferimento (Serie A)</label>
              <select id="seasonStart" class="field-input" [(ngModel)]="seasonStart"
                      [attr.aria-describedby]="'legend-seasonStart'">
                @for (s of seasons(); track s) {
                  <option [value]="s">{{ s }}/{{ s + 1 }}</option>
                }
              </select>
              <app-field-legend
                fieldId="legend-seasonStart"
                [description]="SETUP_LEGENDS['seasonStart'].description"
                [examples]="SETUP_LEGENDS['seasonStart'].examples" />
            </div>

            <p class="section-divider">Partecipanti</p>

            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="numParticipants">Numero di squadre partecipanti</label>
                <input id="numParticipants" class="field-input" type="number" min="2" max="20"
                       [(ngModel)]="numParticipants"
                       (change)="resizeParticipants()"
                       [attr.aria-describedby]="'legend-numParticipants'" />
                <app-field-legend
                  fieldId="legend-numParticipants"
                  [description]="SETUP_LEGENDS['numParticipants'].description"
                  [examples]="SETUP_LEGENDS['numParticipants'].examples" />
              </div>
              <div class="field-group">
                <label class="field-label" for="defaultBudget">Budget iniziale per squadra <span class="field-hint">cr.</span></label>
                <input id="defaultBudget" class="field-input" type="number" min="100" max="2000" step="25"
                       [ngModel]="defaultBudget"
                       (ngModelChange)="defaultBudget = +$event; applyDefaultBudget()"
                       [attr.aria-describedby]="'legend-defaultBudget'" />
                <app-field-legend
                  fieldId="legend-defaultBudget"
                  [description]="SETUP_LEGENDS['defaultBudget'].description"
                  [examples]="SETUP_LEGENDS['defaultBudget'].examples" />
              </div>
            </div>

            <p class="section-divider">Quote rosa per ruolo</p>

            <div class="quota-grid">
              @for (role of allRoles; track role) {
                <div class="field-group">
                  <label class="field-label" [style.color]="roleColor(role)" [attr.for]="'role-' + role">
                    @switch (role) {
                      @case ('P') { Portieri (P) — n° slot in rosa }
                      @case ('D') { Difensori (D) — n° slot in rosa }
                      @case ('C') { Centrocampisti (C) — n° slot in rosa }
                      @case ('A') { Attaccanti (A) — n° slot in rosa }
                    }
                  </label>
                  <input [id]="'role-' + role" class="field-input" type="number" min="1" max="20"
                         [ngModel]="roleQuotas[role] ?? 0"
                         (ngModelChange)="roleQuotas[role] = +$event"
                         [attr.aria-describedby]="'legend-roleQuotas'" />
                </div>
              }
            </div>
            <app-field-legend
              fieldId="legend-roleQuotas"
              [description]="SETUP_LEGENDS['roleQuotas'].description"
              [examples]="SETUP_LEGENDS['roleQuotas'].examples" />

            <p class="section-divider">Modello di inflazione di mercato</p>

            <label class="strategy-check" [class.active]="useInflationBaseline">
              <input type="checkbox" [(ngModel)]="useInflationBaseline" />
              <span>Applica baseline di inflazione basata sul numero di partecipanti</span>
            </label>

            @if (useInflationBaseline) {
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="inflationPct">Soglia percentile inflazione <span class="field-hint">0–1</span></label>
                  <input id="inflationPct" class="field-input" type="number" min="0" max="1" step="0.05"
                         [ngModel]="inflationPercentileThreshold"
                         (ngModelChange)="inflationPercentileThreshold = +$event"
                         [attr.aria-describedby]="'legend-inflationPercentileThreshold'" />
                  <app-field-legend
                    fieldId="legend-inflationPercentileThreshold"
                    [description]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].description"
                    [examples]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="maxInflation">Moltiplicatore massimo inflazione <span class="field-hint">≥1</span></label>
                  <input id="maxInflation" class="field-input" type="number" min="1" max="5" step="0.05"
                         [ngModel]="maxInflationMultiplier"
                         (ngModelChange)="maxInflationMultiplier = +$event"
                         [attr.aria-describedby]="'legend-maxInflationMultiplier'" />
                  <app-field-legend
                    fieldId="legend-maxInflationMultiplier"
                    [description]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].description"
                    [examples]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].examples" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="baseInflationRate">Tasso di inflazione di base <span class="field-hint">0–1</span></label>
                  <input id="baseInflationRate" class="field-input" type="number" min="0" max="1" step="0.01"
                         [ngModel]="baseInflationRate"
                         (ngModelChange)="baseInflationRate = +$event"
                         [attr.aria-describedby]="'legend-baseInflationRate'" />
                  <app-field-legend
                    fieldId="legend-baseInflationRate"
                    [description]="OPTIMIZER_LEGENDS['baseInflationRate'].description"
                    [examples]="OPTIMIZER_LEGENDS['baseInflationRate'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="baselineParticipants">Partecipanti baseline del modello</label>
                  <input id="baselineParticipants" class="field-input" type="number" min="2" max="20" step="1"
                         [ngModel]="baselineParticipants"
                         (ngModelChange)="baselineParticipants = +$event"
                         [attr.aria-describedby]="'legend-baselineParticipants'" />
                  <app-field-legend
                    fieldId="legend-baselineParticipants"
                    [description]="OPTIMIZER_LEGENDS['baselineParticipants'].description"
                    [examples]="OPTIMIZER_LEGENDS['baselineParticipants'].examples" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="teamStrengthMul">Moltiplicatore Elo di Club <span class="field-hint">0–2</span></label>
                  <input id="teamStrengthMul" class="field-input" type="number" min="0" max="2" step="0.05"
                         [ngModel]="teamStrengthMultiplier"
                         (ngModelChange)="teamStrengthMultiplier = +$event"
                         [attr.aria-describedby]="'legend-teamStrengthMultiplier'" />
                  <app-field-legend
                    fieldId="legend-teamStrengthMultiplier"
                    [description]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].description"
                    [examples]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].examples" />
                </div>
              </div>
            }

            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="referenceBudget">Budget di riferimento per il modello <span class="field-hint">cr.</span></label>
                <input id="referenceBudget" class="field-input" type="number" min="1" max="10000" step="50"
                       [(ngModel)]="referenceBudget"
                       [attr.aria-describedby]="'legend-referenceBudget'" />
                <app-field-legend
                  fieldId="legend-referenceBudget"
                  [description]="SETUP_LEGENDS['referenceBudget'].description"
                  [examples]="SETUP_LEGENDS['referenceBudget'].examples" />
              </div>
              <div class="field-group">
                <label class="field-label" for="budgetInitial">Budget effettivo di questa sessione <span class="field-hint">cr.</span></label>
                <input id="budgetInitial" class="field-input" type="number" min="1" max="10000" step="50"
                       [(ngModel)]="budgetInitial"
                       [attr.aria-describedby]="'legend-budgetInitial'" />
                <app-field-legend
                  fieldId="legend-budgetInitial"
                  [description]="SETUP_LEGENDS['budgetInitial'].description"
                  [examples]="SETUP_LEGENDS['budgetInitial'].examples" />
              </div>
            </div>

            <p class="section-divider">Modalità di valutazione del giocatore</p>

            <div class="field-group">
              <label class="field-label" for="valuationMode">Algoritmo usato per stimare il valore del giocatore</label>
              <select id="valuationMode" class="field-input" [(ngModel)]="valuationMode"
                      [attr.aria-describedby]="'legend-valuationMode'">
                <option value="PER_MATCH_RATING">Per-Match Rating (media voto ponderata)</option>
                <option value="SEASON_VALUE">Season Value (valore complessivo di stagione)</option>
              </select>
              <app-field-legend
                fieldId="legend-valuationMode"
                [description]="SETUP_LEGENDS['valuationMode'].description"
                [examples]="SETUP_LEGENDS['valuationMode'].examples" />
            </div>

            <p class="section-divider">Filtro sui giocatori e replacement level</p>

            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="replacementMethod">Replacement level per VAR/ESV</label>
                <select id="replacementMethod" class="field-input" [(ngModel)]="replacementMethod"
                        [attr.aria-describedby]="'legend-replacementMethod'">
                  <option value="percentile">Percentile (bottom-N% per ruolo)</option>
                  <option value="roster_depth">Roster depth (quota rosa per ruolo)</option>
                </select>
                <app-field-legend
                  fieldId="legend-replacementMethod"
                  [description]="OPTIMIZER_LEGENDS['replacementMethod'].description"
                  [examples]="OPTIMIZER_LEGENDS['replacementMethod'].examples" />
              </div>
              <div class="field-group">
                <label class="field-label" for="minStartProb">Soglia minima di titolarità <span class="field-hint">0–1 (vuoto = nessun filtro)</span></label>
                <input id="minStartProb" class="field-input" type="number" min="0" max="1" step="0.05"
                       [ngModel]="minStartProbability"
                       (ngModelChange)="minStartProbability = $event === null || $event === '' ? null : +$event"
                       [attr.aria-describedby]="'legend-minStartProbability'" />
                <app-field-legend
                  fieldId="legend-minStartProbability"
                  [description]="OPTIMIZER_LEGENDS['minStartProbability'].description"
                  [examples]="OPTIMIZER_LEGENDS['minStartProbability'].examples" />
              </div>
            </div>

            <!-- Advanced -->
            <button class="advanced-toggle" (click)="showAdvanced = !showAdvanced">
              Impostazioni avanzate EWMA e spillover {{ showAdvanced ? '▲' : '▼' }}
            </button>

            @if (showAdvanced) {
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="alpha">Fattore EWMA (alpha) <span class="field-hint">0–1</span></label>
                  <input id="alpha" class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="alpha"
                         [attr.aria-describedby]="'legend-alpha'" />
                  <app-field-legend
                    fieldId="legend-alpha"
                    [description]="SETUP_LEGENDS['alpha'].description"
                    [examples]="SETUP_LEGENDS['alpha'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="spilloverAdj">Spillover stesso ruolo, tier adiacente <span class="field-hint">0–1</span></label>
                  <input id="spilloverAdj" class="field-input" type="number" min="0" max="1" step="0.01"
                         [(ngModel)]="spilloverAdj"
                         [attr.aria-describedby]="'legend-spilloverAdj'" />
                  <app-field-legend
                    fieldId="legend-spilloverAdj"
                    [description]="SETUP_LEGENDS['spilloverAdj'].description"
                    [examples]="SETUP_LEGENDS['spilloverAdj'].examples" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="spilloverCross">Spillover cross-ruolo <span class="field-hint">0–1</span></label>
                  <input id="spilloverCross" class="field-input" type="number" min="0" max="1" step="0.01"
                         [(ngModel)]="spilloverCross"
                         [attr.aria-describedby]="'legend-spilloverCross'" />
                  <app-field-legend
                    fieldId="legend-spilloverCross"
                    [description]="SETUP_LEGENDS['spilloverCross'].description"
                    [examples]="SETUP_LEGENDS['spilloverCross'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="lowCostPercentile">Percentile soglia "low-cost" <span class="field-hint">0–1</span></label>
                  <input id="lowCostPercentile" class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="lowCostPercentile"
                         [attr.aria-describedby]="'legend-lowCostPercentile'" />
                  <app-field-legend
                    fieldId="legend-lowCostPercentile"
                    [description]="SETUP_LEGENDS['lowCostPercentile'].description"
                    [examples]="SETUP_LEGENDS['lowCostPercentile'].examples" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="minIndex">Indice EWMA minimo <span class="field-hint">0–1</span></label>
                  <input id="minIndex" class="field-input" type="number" min="0" max="1" step="0.1"
                         [(ngModel)]="minIndex"
                         [attr.aria-describedby]="'legend-minIndex'" />
                  <app-field-legend
                    fieldId="legend-minIndex"
                    [description]="SETUP_LEGENDS['minIndex'].description"
                    [examples]="SETUP_LEGENDS['minIndex'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="maxIndex">Indice EWMA massimo <span class="field-hint">1–5</span></label>
                  <input id="maxIndex" class="field-input" type="number" min="1" max="5" step="0.1"
                         [(ngModel)]="maxIndex"
                         [attr.aria-describedby]="'legend-maxIndex'" />
                  <app-field-legend
                    fieldId="legend-maxIndex"
                    [description]="SETUP_LEGENDS['maxIndex'].description"
                    [examples]="SETUP_LEGENDS['maxIndex'].examples" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label" for="tierLow">Soglia tier basso (LOW) <span class="field-hint">0–1</span></label>
                  <input id="tierLow" class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="tierLow"
                         [attr.aria-describedby]="'legend-tierLow'" />
                  <app-field-legend
                    fieldId="legend-tierLow"
                    [description]="SETUP_LEGENDS['tierLow'].description"
                    [examples]="SETUP_LEGENDS['tierLow'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="tierTop">Soglia tier alto (TOP) <span class="field-hint">0–1</span></label>
                  <input id="tierTop" class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="tierTop"
                         [attr.aria-describedby]="'legend-tierTop'" />
                  <app-field-legend
                    fieldId="legend-tierTop"
                    [description]="SETUP_LEGENDS['tierTop'].description"
                    [examples]="SETUP_LEGENDS['tierTop'].examples" />
                </div>
              </div>
            }

            @if (initError()) {
              <app-error-boundary title="Errore di inizializzazione sessione" [message]="initError()!" />
            }

            <button class="run-btn" (click)="startAuction()" [disabled]="starting() || !seasons().length">
              @if (starting()) {
                <span class="spinner"></span> Avvio in corso…
              } @else {
                Avvia asta
              }
            </button>

            <button class="secondary-btn full-w" (click)="fileInput.click()" [disabled]="starting()"
                    title="Carica un file JSON precedentemente esportato con 'Salva sessione' per riprendere un'asta interrotta.">
              Riprendi da file di salvataggio (.json)
            </button>
            <input #fileInput type="file" accept=".json" style="display:none"
                   (change)="onResumeFile($event)" />

          </aside>

          <!-- Participants editor -->
          <section class="setup-right">
            <div class="card">
              <p class="card-section-label" style="margin-bottom:12px">
                Elenco partecipanti ({{ participants().length }}) — modifica nome e budget individuale
              </p>
              <div class="participants-list">
                <div class="participants-list-header">
                  <span>Nome visualizzato</span><span>Budget iniziale (cr.)</span>
                </div>
                @for (p of participants(); track p.participantId; let i = $index) {
                  <div class="participant-edit-row">
                    <input class="field-input"
                           [ngModel]="p.displayName"
                           (ngModelChange)="updateName(i, $event)"
                           [placeholder]="'Squadra ' + (i + 1)"
                           [attr.aria-label]="'Nome partecipante ' + (i + 1)" />
                    <input class="field-input budget-input" type="number"
                           [ngModel]="p.budgetInitial"
                           (ngModelChange)="updateBudget(i, +$event)"
                           min="100" max="2000" step="25"
                           [attr.aria-label]="'Budget iniziale partecipante ' + (i + 1)" />
                  </div>
                }
              </div>
            </div>
          </section>

        </div>
      </div>
    }
  `,
  styleUrls: ['./auction.component.scss'],
})
export class AuctionComponent {
  private readonly auctionService = inject(AuctionService);
  private readonly quotationService = inject(QuotationService);

  readonly allRoles: readonly AuctionRole[] = AUCTION_ROLES;
  readonly allTiers: readonly AuctionTier[] = ['LOW', 'MID', 'TOP'];

  /** Legende dei campi del pannello di setup (configurazione iniziale). */
  protected readonly SETUP_LEGENDS = SETUP_LEGENDS;
  /** Legende dei campi della vista live (lookup + registrazione). */
  protected readonly LIVE_LEGENDS = LIVE_LEGENDS;

  // ── Setup form state (plain properties — bound via (change) events) ──
  seasonStart = 2024;
  numParticipants = 8;
  defaultBudget = 500;
  showAdvanced = false;
  useInflationBaseline = true;
  referenceBudget = 300;
  budgetInitial = 300;
  roleQuotas: Partial<Record<AuctionRole, number>> = { P: 3, D: 8, C: 8, A: 6 };
  valuationMode: ValuationMode = 'PER_MATCH_RATING';

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
  readonly participants = signal<AuctionParticipantSetup[]>(makeParticipants(8, 500));
  readonly starting = signal(false);
  readonly initError = signal<string | null>(null);

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

  readonly varRanking = signal<VarRankingItem[]>([]);
  readonly varLoading = signal(false);

  /**
   * Ordinamento della tabella "Migliori affari".
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
  lookupQuery = '';    // display text in lookup input
  lookupId = '';       // resolved playerId
  recordPlayerId = '';
  recordPlayerName = ''; // display text in record input
  recordWinnerId = '';
  recordPrice = 1;

  // ── Initial budgets map (for budget-bar computation) ──────────────────
  private readonly initialBudgets = new Map<string, number>();

  readonly reversedAssignments = computed(() =>
    [...(this.summary()?.assignments ?? [])].reverse(),
  );

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
      case 'name': return item.name.toLowerCase();
      case 'role': return item.role;
      case 'esv': return item.esv;
      case 'expectedPrice': return item.expectedPrice;
      case 'seasonValue': return item.seasonValue;
      case 'startProbability': return item.startProbability;
      case 'buySignal': return item.buySignal ? 1 : 0;
    }
  }

  constructor() {
    this.quotationService.getSeasons().subscribe({
      next: s => {
        const sorted = [...s].sort((a, b) => b - a);
        this.seasons.set(sorted);
        if (sorted.length) this.seasonStart = sorted[0];
      },
      error: () => this.seasons.set([2025, 2024, 2023]),
    });

    // Pool autocomplete: debounce query → call pool endpoint
    this.poolQuery$.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      switchMap(q => {
        const sid = this.sessionId();
        if (!sid) return [];
        return this.auctionService.pool(sid, q);
      }),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: items => { this.poolSuggestions.set(items); this.poolOpen.set(items.length > 0); },
      error: () => { this.poolSuggestions.set([]); this.poolOpen.set(false); },
    });
  }

  // ── Setup helpers ─────────────────────────────────────────────────────

  resizeParticipants(): void {
    this.participants.set(makeParticipants(this.numParticipants, this.defaultBudget, this.participants()));
  }

  updateName(i: number, name: string): void {
    this.participants.update(arr => {
      const next = [...arr];
      next[i] = { ...next[i], displayName: name };
      return next;
    });
  }

  updateBudget(i: number, budget: number): void {
    this.participants.update(arr => {
      const next = [...arr];
      next[i] = { ...next[i], budgetInitial: budget };
      return next;
    });
  }

  /** Propaga "Budget each" a tutti i participants. */
  applyDefaultBudget(): void {
    this.participants.update(arr =>
      arr.map(p => ({ ...p, budgetInitial: this.defaultBudget })),
    );
  }

  // ── Session init ──────────────────────────────────────────────────────

  startAuction(): void {
    this.starting.set(true);
    this.initError.set(null);
    this._cacheInitialBudgets(this.participants());

    this.auctionService.init({
      seasonStart: this.seasonStart,
      participants: this.participants(),
      config: {
        numParticipants: this.numParticipants,
        roleQuotas: this.roleQuotas,
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
        ...(this.useInflationBaseline ? {
          inflationConfig: {
            inflationPercentileThreshold: this.inflationPercentileThreshold,
            maxInflationMultiplier: this.maxInflationMultiplier,
            baseInflationRate: this.baseInflationRate,
            baselineParticipants: this.baselineParticipants,
            teamStrengthMultiplier: this.teamStrengthMultiplier,
          },
        } : {}),
        // Pool pre-filter and replacement level are always sent so the
        // server-side VarEngine applies them regardless of the inflation
        // toggle (they affect the ranking, not the price model).
        minStartProbability: this.minStartProbability,
        replacementMethod: this.replacementMethod,
        referenceBudget: this.referenceBudget,
        budgetInitial: this.budgetInitial,
        valuationMode: this.valuationMode,
      },
    }).subscribe({
      next: res => {
        this.sessionId.set(res.sessionId);
        this.starting.set(false);
        this.refreshSummary();
      },
      error: err => {
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
          next: res => {
            this.sessionId.set(res.sessionId);
            this.starting.set(false);
            this.refreshSummary();
          },
          error: err => {
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
    if (!q.trim()) { this.poolSuggestions.set([]); this.poolOpen.set(false); }
  }

  selectPoolPlayer(p: AuctionPlayerSummary): void {
    this.lookupId = p.playerId;
    this.lookupQuery = p.name;
    // pre-fill record card too
    this.recordPlayerId = p.playerId;
    this.recordPlayerName = `${p.name} (${p.role} · ${p.realTeam})`;
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
      alt: this.auctionService.alternatives(sid, playerId),
    }).subscribe({
      next: ({ proj, alt }) => {
        this.projection.set(proj);
        this.altResult.set(alt);
        this.lookupLoading.set(false);
      },
      error: err => {
        this.lookupError.set(err.error?.detail ?? 'Player not found');
        this.lookupLoading.set(false);
      },
    });
  }

  submitRecord(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.recordLoading.set(true);
    this.recordError.set(null);
    this.recordRejectionCode.set(null);

    this.auctionService.record(sid, {
      playerId: this.recordPlayerId,
      winnerParticipantId: this.recordWinnerId,
      finalPrice: this.recordPrice,
    }).subscribe({
      next: res => {
        if (!res.success) {
          this.recordError.set(res.rejectionReason ?? 'Assignment rejected');
          this.recordRejectionCode.set(res.rejectionCode ?? null);
        } else {
          this.recordPlayerId = '';
          this.recordPlayerName = '';
          this.recordWinnerId = '';
          this.recordPrice = 1;
          this.recordError.set(null);
          this.projection.set(null);
          this.altResult.set(null);
          this.lookupId = '';
          this.lookupQuery = '';
          this.refreshSummary();
        }
        this.recordLoading.set(false);
      },
      error: err => {
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
      next: () => { this.undoLoading.set(false); this.refreshSummary(); },
      error: () => this.undoLoading.set(false),
    });
  }

  saveToFile(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.auctionService.serialize(sid).subscribe({
      next: res => {
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
      next: s => {
        this.summary.set(s);
        this.summaryLoading.set(false);
        // Populate initialBudgets from setup participants if not already set
        if (this.initialBudgets.size === 0) {
          s.participants.forEach(p => {
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
      next: res => { this.varRanking.set(res.items); this.varLoading.set(false); },
      error: () => this.varLoading.set(false),
    });
  }

  // ── Template helpers ──────────────────────────────────────────────────

  roleColor(role: string): string { return ROLE_COLOR[role] ?? 'var(--color-text-secondary)'; }
  tierColor(tier: AuctionTier): string { return TIER_COLOR[tier]; }

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
    return this.summary()?.participants.find(p => p.participantId === participantId)?.displayName
      ?? participantId;
  }

  // ── Private ───────────────────────────────────────────────────────────

  private _cacheInitialBudgets(participants: AuctionParticipantSetup[]): void {
    this.initialBudgets.clear();
    participants.forEach(p => this.initialBudgets.set(p.participantId, p.budgetInitial));
  }

  private _resetLiveState(): void {
    this.sessionId.set(null);
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
