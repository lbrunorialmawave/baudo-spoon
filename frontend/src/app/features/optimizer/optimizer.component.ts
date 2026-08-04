import {
  Component, computed, inject, signal, DestroyRef,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, PercentPipe } from '@angular/common';
import {
  catchError, finalize, interval, map, of, startWith, switchMap, take, takeWhile, throwError,
} from 'rxjs';
import { OptimizerService } from '../../core/services/optimizer.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  FormationConfig,
  MultiStrategyResult,
  OptimizationRequest,
  OptimizationResult,
  SquadPlayer,
  StrategyProfile,
} from '../../core/models/api.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';
import { FieldLegendComponent, FieldLegendExample } from '../../shared/components/field-legend/field-legend.component';
import {
  findOptimizerPreset,
  OPTIMIZER_PRESET_NONE,
  OPTIMIZER_PRESETS,
  OptimizerPreset,
} from '../../core/constants/optimizer-presets';
import { OptimizerPlayerDrawerComponent } from './optimizer-player-drawer/optimizer-player-drawer.component';

const STRATEGY_META: Record<string, { label: string; icon: string }> = {
  BALANCED:        { label: 'Bilanciata',      icon: '⚖️' },
  SUPER_DEFENSIVE: { label: 'Super-difensiva', icon: '🛡️' },
  SUPER_OFFENSIVE: { label: 'Super-offensiva', icon: '⚡' },
  MIXED:           { label: 'Mista',           icon: '🎯' },
};

const ROLE_COLORS: Record<string, string> = {
  P: 'var(--color-role-gk)',
  D: 'var(--color-role-def)',
  C: 'var(--color-role-mid)',
  A: 'var(--color-role-fwd)',
};

const ROLE_LABELS: Record<string, string> = { P: 'GK', D: 'DEF', C: 'MID', A: 'FWD' };

/** Above this N with MC enabled, UI uses POST /optimize/jobs + polling. */
const ASYNC_MC_THRESHOLD = 25;
const JOB_POLL_MS = 1500;
const JOB_MAX_POLLS = 120; // ~3 min


const ALL_FORMATIONS: FormationConfig[] = [
  { label: '3-4-3', defenders: 3, midfielders: 4, forwards: 3 },
  { label: '3-5-2', defenders: 3, midfielders: 5, forwards: 2 },
  { label: '4-3-3', defenders: 4, midfielders: 3, forwards: 3 },
  { label: '4-4-2', defenders: 4, midfielders: 4, forwards: 2 },
  { label: '4-5-1', defenders: 4, midfielders: 5, forwards: 1 },
  { label: '5-3-2', defenders: 5, midfielders: 3, forwards: 2 },
];

/**
 * Legende minuziose dei campi del configuratore dell'Optimizer.
 * Ogni entry spiega cosa fa il campo e come viene utilizzato dal risolutore
 * ILP, con esempi concreti d'uso per facilitare la comprensione a tutti.
 */
/**
 * Map of field legends used across the optimizer and the auction screens.
 * Exported so the auction screen can reuse identical copy for the shared
 * fields (inflation tuning, replacement level, start-probability pre-filter,
 * Club Elo team-strength multiplier) without duplicating text.
 */
export const OPTIMIZER_LEGENDS: Readonly<Record<string, { description: string; examples: readonly FieldLegendExample[] }>> = {
  seasonStart: {
    description: 'Chiave di lookup sul database: seleziona la stagione da cui caricare listini (qt_a), id-map e predizioni ML. Non entra nella funzione obiettivo; determina solo quale pool di giocatori viene costruito.',
    examples: [
      { label: '2025', value: 'stagione 2025/26 (corrente)' },
      { label: '2024', value: 'stagione 2024/25 (storica)' },
    ],
  },
  budget: {
    description: 'Vincolo hard del solver: la somma dei costi effettivi dei 25 giocatori scelti non può superare questo valore in crediti. È il budget totale della rosa, non il budget per ruolo.',
    examples: [
      { label: '300 cr.', value: 'listino ufficiale storico' },
      { label: '500 cr.', value: 'lega moderna tipica' },
      { label: '1000 cr.', value: 'lega ad alto potere d\'acquisto' },
    ],
  },
  numParticipants: {
    description: 'Numero di squadre in lega. Entra nel modello di inflazione del costo effettivo: per ogni partecipante oltre baselineParticipants cresce il moltiplicatore sui giocatori sopra la soglia di percentile. Non cambia le quote di rosa.',
    examples: [
      { label: '4', value: 'poca competizione → inflazione bassa' },
      { label: '8', value: 'default classico' },
      { label: '10–12', value: 'alta competizione → costi effettivi più alti' },
    ],
  },
  minQtA: {
    description: 'Filtro pre-solver sul pool: restano solo i giocatori con listino (qt_a) ≥ questa soglia. Non è un minimo di spesa per ruolo né una quota di rosa. Default 1 = esclude listino 0 / non quotati.',
    examples: [
      { label: '0', value: 'include anche listino 0' },
      { label: '1', value: 'default: solo giocatori quotati (≥ 1 cr.)' },
      { label: '5', value: 'pool ristretto ai listini ≥ 5 cr.' },
    ],
  },
  solverTimeoutSeconds: {
    description: 'Limite di tempo del solver ILP (PuLP/CBC). Allo scadere restituisce la migliore soluzione ammissibile trovata, o TIMEOUT se non ne ha ancora una. Non altera la funzione obiettivo né i vincoli.',
    examples: [
      { label: '15 s', value: 'UI reattiva, pool semplice' },
      { label: '30 s', value: 'default bilanciato' },
      { label: '60 s', value: 'pool grande / MANTRA / molti vincoli' },
    ],
  },
  minDistinctTeams: {
    description: 'Vincolo hard: nella rosa devono comparire almeno N club di Serie A distinti (campo real_team). Riduce la concentrazione su pochi club; se troppo alto rispetto al pool può rendere il problema infeasible.',
    examples: [
      { label: '8', value: 'diversificazione leggera' },
      { label: '12', value: 'default' },
      { label: '16', value: 'molto restrittivo' },
    ],
  },
  maxPlayersPerTeam: {
    description: 'Vincolo hard: al massimo K giocatori con lo stesso real_team. Impedisce di costruire la rosa intorno a un solo club.',
    examples: [
      { label: '2', value: 'molto stretto' },
      { label: '4', value: 'default' },
      { label: '6', value: 'concentrazione alta consentita' },
    ],
  },
  bigTeamsCap: {
    description: 'Vincolo hard sul conteggio totale di giocatori il cui real_team è in bigTeams. È un tetto aggregato (non per singola big); si somma al maxPlayersPerTeam.',
    examples: [
      { label: '6', value: 'rosa poco dipendente dalle big' },
      { label: '10', value: 'default API' },
      { label: '15', value: 'tetto largo' },
    ],
  },
  bigTeams: {
    description: 'Insieme di nomi club usati dal vincolo bigTeamsCap. I nomi devono coincidere esattamente con real_team nel database (es. "Inter", non "FC Internazionale").',
    examples: [
      { label: 'Inter, Milan, Juventus, Napoli', value: 'default top 4' },
      { label: '+ Roma, Lazio, Atalanta', value: 'top 7 allargato' },
    ],
  },
  maxSinglePlayerBudgetShare: {
    description: 'Vincolo hard sul costo effettivo: nessun giocatore può costare più di share × budget (es. 0.30 × 500 = 150 cr.). Evita che il solver concentri troppo budget su un solo nominativo.',
    examples: [
      { label: '0.22', value: 'max ~110 cr. su 500' },
      { label: '0.30', value: 'default: max 30% del budget' },
      { label: '0.40', value: 'ammessi top molto costosi' },
    ],
  },
  mustInclude: {
    description: 'Vincolo hard: questi player_id devono comparire nella soluzione. Se incompatibili con budget/quote il solver fallisce (INFEASIBLE).',
    examples: [
      { label: 'fm-12345', value: 'forza un giocatore' },
      { label: '(vuoto)', value: 'nessun vincolo' },
    ],
  },
  exclude: {
    description: 'Vincolo hard: questi player_id sono rimossi dal pool decisionale prima/durante l\'ILP. Utile per infortuni, squalifiche o preferenze personali.',
    examples: [
      { label: 'fm-67890', value: 'escludi uno' },
      { label: 'fm-1, fm-2', value: 'escludi più ID' },
    ],
  },
  ruleset: {
    description: 'Cambia il modello di assegnazione ruoli. CLASSIC: 4 ruoli (P/D/C/A) con quote 3/8/8/6. MANTRA: 12 ruoli multi-slot (eligible_roles) e quote Por/Dc/B/… che devono sommare a 25.',
    examples: [
      { label: 'CLASSIC', value: 'Fantacalcio tradizionale' },
      { label: 'MANTRA', value: 'ruoli modulari e multi-ruolo' },
    ],
  },
  riskAversion: {
    description: 'Coefficiente nella funzione obiettivo: effective_score = … − riskAversion × prediction_std. A 0 il solver è risk-neutral. Valori > 0 penalizzano giocatori con alta volatilità dell\'ensemble. Se prediction_std manca, la leva non ha effetto.',
    examples: [
      { label: '0.0', value: 'neutrale (default): max score atteso' },
      { label: '0.5–1.0', value: 'penalità moderata alla varianza' },
      { label: '1.2–1.5', value: 'floor alto, preferisce titolari stabili' },
    ],
  },

  monteCarloEnabled: {
    description: 'Attiva robustezza Monte Carlo sullo score. Default off = ILP deterministico (legacy). Con SAA il solver riesegue N scenari con score campionati da residuali / prediction_std e restituisce frequenza di selezione e stability index.',
    examples: [
      { label: 'off', value: 'path classico, 1× ILP' },
      { label: 'on + mean_std', value: '1× ILP su mean − λ·std' },
      { label: 'on + saa_frequency', value: 'N× ILP; ranking per frequenza' },
    ],
  },
  monteCarloMode: {
    description: 'mean_std: singolo solve risk-adjusted. saa_frequency: N scenari; rosa rappresentativa + selection frequency. Non combinare SAA aggressivo con riskAversion alto senza leggere entrambi gli effetti.',
    examples: [
      { label: 'mean_std', value: 'latenza ~1×, buon default prudente' },
      { label: 'saa_frequency', value: 'latenza ~N×; N≤10 sync consigliato' },
    ],
  },
  nSimulations: {
    description: 'Numero scenari SAA (o campioni per mean_std). Sync tipico 5–25; oltre ~50 preferire job async. Cap server: API_OPTIMIZER_MAX_SIMULATIONS.',
    examples: [
      { label: '5–10', value: 'UI reattiva / MANTRA' },
      { label: '25–50', value: 'CLASSIC sync accettabile' },
      { label: '100+', value: 'solo async /jobs' },
    ],
  },
  nearOptimal: {
    description: 'Dopo la rosa ottima, esclude i top-M scorer e ri-ottimizza per alternative vicine in score. Utile in asta quando i top vengono comprati da altri.',
    examples: [
      { label: 'off', value: 'solo rosa primaria' },
      { label: 'on, top-2, 3 alt', value: 'default sensato' },
    ],
  },
  varBlend: {
    description: 'Peso in [0,1] del VAR nella funzione obiettivo: (1−varBlend)×base_metric + varBlend×var_score. 0 = solo projected_score/season_value; 1 = puro Value Above Replacement.',
    examples: [
      { label: '0.0', value: 'VAR disattivato (default)' },
      { label: '0.25–0.35', value: 'blend moderato' },
      { label: '1.0', value: 'obiettivo interamente VAR' },
    ],
  },
  esvWeight: {
    description: 'Peso additivo dell\'ESV (Expected Surplus Value, segnale "affare") nella funzione obiettivo: … + esvWeight × esv. 0 = disattivato. Premia giocatori sotto-prezzati rispetto al contributo atteso; non è il season_value grezzo.',
    examples: [
      { label: '0', value: 'ESV off (default)' },
      { label: '0.15–0.25', value: 'bias value moderato' },
      { label: '0.40–0.50', value: 'caccia aggressiva ai bargain' },
    ],
  },
  valuationMode: {
    description: 'Sceglie la metrica base (base_metric) nella funzione obiettivo. PER_MATCH_RATING usa projected_score (fantavoto/partita). SEASON_VALUE usa season_value (rating × presenze attese). Non cambia i vincoli di rosa.',
    examples: [
      { label: 'PER_MATCH_RATING', value: 'default: rendimento a partita' },
      { label: 'SEASON_VALUE', value: 'valore totale di stagione proiettato' },
    ],
  },
  replacementMethod: {
    description: 'Come si calcola il replacement level usato da VAR/ESV quando varBlend o esvWeight > 0. "percentile" = basso percentile per ruolo nel pool; "roster_depth" = soglia legata a numParticipants × quota ruolo.',
    examples: [
      { label: 'percentile', value: 'default: bottom percentile per ruolo' },
      { label: 'roster_depth', value: 'profondità rosa di lega' },
    ],
  },
  minStartProbability: {
    description: 'Filtro pre-ILP: i giocatori con start_probability < soglia sono esclusi dal pool passato al solver. null/vuoto = nessun filtro. Quando varBlend/esvWeight > 0 la stessa soglia è allineata al motore VAR.',
    examples: [
      { label: 'vuoto', value: 'nessun filtro (default)' },
      { label: '0.60', value: 'taglia riserve chiare' },
      { label: '0.75', value: 'solo alta titolarità' },
    ],
  },
  formations: {
    description: 'Moduli valutati post-soluzione in formation_feasibility (✓/✗). Non sono tutti vincoli hard: solo preferredFormation, se impostata, è un vincolo del solver. Max 6 moduli lato API.',
    examples: [
      { label: '4-3-3', value: 'classico offensivo' },
      { label: '4-4-2', value: 'equilibrato' },
      { label: '3-5-2', value: 'centrocampo largo' },
    ],
  },
  preferredFormation: {
    description: 'Unico modulo imposto come vincolo hard all\'ILP (la rosa deve poter schierare difensori/centrocampisti/attaccanti di quel modulo). Le altre formations restano solo check informativi. Vuoto = nessun vincolo di modulo.',
    examples: [
      { label: 'Nessuna', value: 'nessun vincolo hard di modulo' },
      { label: '4-3-3', value: 'forza schierabilità 4-3-3' },
    ],
  },
  inflationPercentileThreshold: {
    description: 'Soglia di percentile di ruolo sotto la quale il costo effettivo resta uguale al listino. Sopra soglia parte l\'inflazione (cresce con distanza dalla soglia e con i partecipanti extra rispetto a baselineParticipants). Default codice 0.7.',
    examples: [
      { label: '0.70', value: 'default: top ~30% del ruolo si gonfiano' },
      { label: '0.85', value: 'inflazione solo sui rarissimi' },
      { label: '0.55', value: 'inflazione più ampia sul listino' },
    ],
  },
  maxInflationMultiplier: {
    description: 'Cap del moltiplicatore costo_effettivo / listino (deve essere ≥ 1.0). Limita quanto può gonfiarsi un top rispetto al qt_a. Il solver usa i costi effettivi nel vincolo di budget.',
    examples: [
      { label: '1.0', value: 'nessuna inflazione (moltiplicatore piatto)' },
      { label: '1.6', value: 'default moderato' },
      { label: '1.9', value: 'mercato caldo' },
    ],
  },
  baseInflationRate: {
    description: 'Tasso base usato dalla curva di inflazione per i partecipanti oltre baselineParticipants (insieme al percentile sopra soglia). Non è un +X% piatto su tutto il listino sotto soglia: sotto threshold il costo resta nominale.',
    examples: [
      { label: '0.03', value: 'mercato freddo' },
      { label: '0.05', value: 'default' },
      { label: '0.08', value: 'alta pressione d\'asta' },
    ],
  },
  baselineParticipants: {
    description: 'Soglia di partecipanti "senza extra-inflazione". Solo i partecipanti oltre questo numero alimentano la parte di curva legata a baseInflationRate. Di solito allineato a numParticipants della lega reale.',
    examples: [
      { label: '8', value: 'default' },
      { label: '6', value: 'simula lega più piccola come baseline' },
      { label: '10', value: 'baseline più alta → meno extra-inflazione a parità di numParticipants' },
    ],
  },
  teamStrengthMultiplier: {
    description: 'Peso ≥ 0 sull\'aggiustamento Elo di club: effective_cost × (1 + weight × elo_normalizzato). 0 = disattivato. Aumenta il costo effettivo dei giocatori di squadre forti (più contendibili in asta), non modifica lo score obiettivo.',
    examples: [
      { label: '0.0', value: 'off (default)' },
      { label: '0.10–0.20', value: 'premio moderato alle big' },
      { label: '0.35–0.40', value: 'forte bias di costo sulle big' },
    ],
  },
  customWeights: {
    description: 'roleWeight della StrategyProfile: moltiplicano il contributo del giocatore in obiettivo per ruolo (P/D/C/A). Usati solo con una strategia selezionata in modalità custom. Non sono quote di rosa né minimi di spesa.',
    examples: [
      { label: '1 / 1 / 1 / 1', value: 'neutro (BALANCED)' },
      { label: '1.2 / 1.3 / 1 / 0.8', value: 'bias difensivo' },
      { label: '0.8 / 0.9 / 1.15 / 1.3', value: 'bias offensivo' },
    ],
  },
};

@Component({
  selector: 'app-optimizer',
  standalone: true,
  imports: [FormsModule, DecimalPipe, PercentPipe, SkeletonComponent, ErrorBoundaryComponent, FieldLegendComponent, OptimizerPlayerDrawerComponent],
  template: `
    <div class="optimizer-root">

      <!-- HEADER -->
      <div class="header">
        <div class="header-left">
          <h1 class="header-title">Ottimizzatore Rosa</h1>
          <p class="header-subtitle">Costruisci la miglior squadra con ILP e robustezza Monte Carlo</p>
        </div>
        <button class="header-run-btn" (click)="run()" [disabled]="running() || !canRun()"
                aria-label="Esegui ottimizzatore">
          @if (running()) { <span class="spinner"></span> }
          {{ running() ? 'Calcolo...' : '⚡ Esegui' }}
        </button>
      </div>

      <!-- MAIN LAYOUT -->
      <div class="main-grid">

        <!-- CONFIG PANEL (scrollable aside) -->
        <aside class="config-panel">
          <!-- Preset picker always visible -->
          <details class="config-group" open>
            <summary class="config-group-title">🎯 Profilo strategico</summary>
            <div class="config-body">
              <label class="field-label" for="opt-preset">Preset (precompila tutto)</label>
              <select id="opt-preset" class="field-input"
                      [ngModel]="selectedPresetId()"
                      (ngModelChange)="onPresetChange($event)">
                <option [ngValue]="OPTIMIZER_PRESET_NONE">Personalizzato</option>
                @for (p of presets; track p.id) {
                  <option [ngValue]="p.id">{{ p.labelIt }} — {{ p.name }}</option>
                }
              </select>
              @if (activePreset(); as preset) {
                <p class="preset-desc">{{ preset.description }}</p>
              } @else {
                <p class="preset-desc muted">Scegli un profilo per impostare rapidamente vincoli, rischio e strategie. Stagione e include/exclude restano sotto il tuo controllo.</p>
              }
            </div>
          </details>

          <!-- Pool & Budget -->
          <details class="config-group" open>
            <summary class="config-group-title">🏊 Pool e budget</summary>
            <div class="config-body">
              <div class="field-row">
                <label class="field-label" for="opt-seasonStart">Stagione</label>
                @if (seasonsLoading()) { <app-skeleton height="36px" /> }
                @else {
                  <select id="opt-seasonStart" class="field-input" [(ngModel)]="seasonStart">
                    @for (s of seasons(); track s) { <option [value]="s">{{ s }}/{{ s + 1 }}</option> }
                  </select>
                }
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-budget">Budget (cr.)</label>
                <input id="opt-budget" class="field-input" type="number" min="200" max="1000" step="25" [(ngModel)]="budget" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-numParticipants">Partecipanti</label>
                <input id="opt-numParticipants" class="field-input" type="number" min="4" max="16" step="1" [(ngModel)]="numParticipants" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-minQtA">Listino minimo (qt_a ≥)</label>
                <input id="opt-minQtA" class="field-input" type="number" min="0" max="10" step="1" [(ngModel)]="minQtA" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-solverTimeout">Timeout solver (s)</label>
                <input id="opt-solverTimeout" class="field-input" type="number" min="5" max="300" step="5" [(ngModel)]="solverTimeoutSeconds" />
              </div>
            </div>
          </details>

          <!-- Vincoli rosa -->
          <details class="config-group">
            <summary class="config-group-title">⛓️ Vincoli rosa</summary>
            <div class="config-body">
              <div class="field-row">
                <label class="field-label" for="opt-minDistinctTeams">Club distinti min</label>
                <input id="opt-minDistinctTeams" class="field-input" type="number" min="1" max="25" step="1" [(ngModel)]="minDistinctTeams" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-maxPlayersPerTeam">Max per club</label>
                <input id="opt-maxPlayersPerTeam" class="field-input" type="number" min="1" max="10" step="1" [(ngModel)]="maxPlayersPerTeam" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-bigTeamsCap">Max big team</label>
                <input id="opt-bigTeamsCap" class="field-input" type="number" min="0" max="25" step="1" [(ngModel)]="bigTeamsCap" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-maxShare">Max budget per giocatore (fraz.)</label>
                <input id="opt-maxShare" class="field-input" type="number" min="0.05" max="1" step="0.05" [(ngModel)]="maxSinglePlayerBudgetShare" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-bigTeamsRaw">Big teams (nomi club)</label>
                <textarea id="opt-bigTeamsRaw" class="field-input field-textarea" rows="2" [(ngModel)]="bigTeamsRaw" placeholder="Inter, Milan..."></textarea>
              </div>
            </div>
          </details>

          <!-- Include / Exclude -->
          <details class="config-group">
            <summary class="config-group-title">✅ Include / ❌ Exclude</summary>
            <div class="config-body">
              <div class="field-row">
                <label class="field-label" for="opt-mustInclude">Must‑include</label>
                <textarea id="opt-mustInclude" class="field-input field-textarea" rows="2" [(ngModel)]="mustIncludeRaw" placeholder="fm-12345, fm-67890"></textarea>
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-exclude">Exclude</label>
                <textarea id="opt-exclude" class="field-input field-textarea" rows="2" [(ngModel)]="excludeRaw" placeholder="fm-12345, fm-67890"></textarea>
              </div>
            </div>
          </details>

          <!-- Obiettivo & Rischio -->
          <details class="config-group">
            <summary class="config-group-title">🎯 Funzione obiettivo</summary>
            <div class="config-body">
              <div class="field-row">
                <label class="field-label" for="opt-ruleset">Ruleset</label>
                <select id="opt-ruleset" class="field-input" [(ngModel)]="ruleset">
                  <option value="CLASSIC">CLASSIC (4 ruoli)</option>
                  <option value="MANTRA">MANTRA (12 ruoli)</option>
                </select>
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-riskAversion">Risk aversion</label>
                <input id="opt-riskAversion" class="field-input" type="number" min="0" max="5" step="0.1" [(ngModel)]="riskAversion" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-varBlend">VAR blend</label>
                <input id="opt-varBlend" class="field-input" type="number" min="0" max="1" step="0.1" [(ngModel)]="varBlend" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-esvWeight">ESV weight</label>
                <input id="opt-esvWeight" class="field-input" type="number" min="0" max="5" step="0.1" [(ngModel)]="esvWeight" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-valuationMode">Metrica base</label>
                <select id="opt-valuationMode" class="field-input" [(ngModel)]="valuationMode">
                  <option value="PER_MATCH_RATING">Per partita</option>
                  <option value="SEASON_VALUE">Valore stagionale</option>
                </select>
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-minStartProb">Prob. titolarità minima</label>
                <input id="opt-minStartProb" class="field-input" type="number" min="0" max="1" step="0.05"
                       [ngModel]="minStartProbability()"
                       (ngModelChange)="minStartProbability.set($event === '' ? null : +$event)" />
              </div>
            </div>
          </details>

          <!-- Monte Carlo -->
          <details class="config-group">
            <summary class="config-group-title">🎲 Monte Carlo</summary>
            <div class="config-body">
              <label class="toggle-row">
                <input type="checkbox" [ngModel]="monteCarloEnabled()" (ngModelChange)="monteCarloEnabled.set($event)" />
                <span>Abilita</span>
              </label>
              @if (monteCarloEnabled()) {
                <div class="field-row">
                  <label class="field-label" for="opt-mc-mode">Mode</label>
                  <select id="opt-mc-mode" class="field-input" [ngModel]="monteCarloMode()" (ngModelChange)="monteCarloMode.set($event)">
                    <option value="saa_frequency">SAA frequency</option>
                    <option value="mean_std">mean – λ·std</option>
                  </select>
                </div>
                <div class="field-row">
                  <label class="field-label" for="opt-mc-n">N simulazioni</label>
                  <input id="opt-mc-n" class="field-input" type="number" min="1" max="200" step="1"
                         [ngModel]="nSimulations()" (ngModelChange)="nSimulations.set(+$event)" />
                </div>
                @if (monteCarloMode() === 'mean_std') {
                  <div class="field-row">
                    <label class="field-label" for="opt-mc-lambda">λ rischio</label>
                    <input id="opt-mc-lambda" class="field-input" type="number" min="0" max="3" step="0.1"
                           [ngModel]="riskLambda()" (ngModelChange)="riskLambda.set(+$event)" />
                  </div>
                }
                @if (riskAversion() > 0) {
                  <div class="warning">⚠️ riskAversion={{ riskAversion() }} + MC attivi: evita doppia penalizzazione.</div>
                }
              }
              <label class="toggle-row">
                <input type="checkbox" [ngModel]="nearOptimalEnabled()" (ngModelChange)="nearOptimalEnabled.set($event)" />
                <span>Alternative near‑optimal</span>
              </label>
            </div>
          </details>

          <!-- Moduli -->
          <details class="config-group">
            <summary class="config-group-title">📐 Moduli</summary>
            <div class="config-body">
              <div class="chip-grid">
                @for (f of allFormations; track f.label) {
                  <label class="chip" [class.active]="selectedFormations().has(f.label)">
                    <input type="checkbox" [checked]="selectedFormations().has(f.label)" (change)="toggleFormation(f.label)" />
                    {{ f.label }}
                  </label>
                }
              </div>
              <div class="field-row" style="margin-top:8px">
                <label class="field-label" for="opt-preferredFormation">Vincolo hard</label>
                <select id="opt-preferredFormation" class="field-input" [(ngModel)]="preferredFormationLabel">
                  <option value="">Nessuno</option>
                  @for (f of allFormations; track f.label) {
                    <option [value]="f.label">{{ f.label }}</option>
                  }
                </select>
              </div>
            </div>
          </details>

          <!-- Costo effettivo -->
          <details class="config-group">
            <summary class="config-group-title">💰 Costo effettivo (inflazione)</summary>
            <div class="config-body">
              <div class="field-row">
                <label class="field-label" for="opt-inflationPercentile">Soglia percentile</label>
                <input id="opt-inflationPercentile" class="field-input" type="number" min="0" max="1" step="0.05" [(ngModel)]="inflationPercentileThreshold" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-maxInflation">Cap moltiplicatore</label>
                <input id="opt-maxInflation" class="field-input" type="number" min="1" max="5" step="0.1" [(ngModel)]="maxInflationMultiplier" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-baseRate">Tasso base</label>
                <input id="opt-baseRate" class="field-input" type="number" min="0" max="1" step="0.01" [(ngModel)]="baseInflationRate" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-baselinePart">Baseline partecipanti</label>
                <input id="opt-baselinePart" class="field-input" type="number" min="2" max="20" step="1" [(ngModel)]="baselineParticipants" />
              </div>
              <div class="field-row">
                <label class="field-label" for="opt-teamStrengthMul">Peso Elo club</label>
                <input id="opt-teamStrengthMul" class="field-input" type="number" min="0" max="2" step="0.05" [(ngModel)]="teamStrengthMultiplier" />
              </div>
            </div>
          </details>

          <!-- Strategie -->
          <details class="config-group">
            <summary class="config-group-title">📋 Strategie</summary>
            <div class="config-body">
              <div class="strategy-list">
                @for (s of availableStrategies(); track s) {
                  <label class="strategy-chip" [class.active]="selectedStrategies().has(s)">
                    <input type="checkbox" [checked]="selectedStrategies().has(s)" (change)="toggleStrategy(s)" />
                    <span>{{ meta(s).icon }} {{ meta(s).label }}</span>
                  </label>
                }
              </div>
              @if (singleStrategySelected()) {
                <button class="text-btn" (click)="showCustomWeights.set(!showCustomWeights())">
                  Pesi ruolo {{ showCustomWeights() ? '▲' : '▼' }}
                </button>
                @if (showCustomWeights()) {
                  <div class="weights-panel">
                    @for (role of ['P','D','C','A']; track role) {
                      <div class="weight-row">
                        <span>{{ roleLabel(role) }}</span>
                        <input type="range" min="0.1" max="3" step="0.05"
                               [ngModel]="customWeights()[role]" (ngModelChange)="setCustomWeight(role, $event)" />
                        <span class="weight-value">{{ customWeights()[role] | number:'1.2-2' }}</span>
                      </div>
                    }
                  </div>
                }
              }
            </div>
          </details>

          <!-- Error & Run button inside config -->
          @if (error()) {
            <app-error-boundary title="Errore" [message]="error()!" />
          }
          <button class="run-btn" (click)="run()" [disabled]="running() || !canRun()">
            @if (running()) { <span class="spinner"></span> }
            {{ running() ? 'Ottimizzazione...' : 'Esegui ottimizzatore' }}
            @if (monteCarloEnabled() && nSimulations() > 25) { <small>(async)</small> }
          </button>
          @if (usedAsyncJob() && jobStatus()) {
            <p class="job-status">Stato job: {{ jobStatus() }}</p>
          }
        </aside>

        <!-- RESULTS PANEL -->
        <section class="results-panel">
          @if (!results() && !running()) {
            <div class="empty-state">
              <div class="empty-icon">🏗️</div>
              <h2 class="empty-title">Nessuna rosa ancora generata</h2>
              <p class="empty-desc">Configura i parametri a sinistra e avvia l'ottimizzatore per vedere la tua formazione ideale.</p>
            </div>
          }

          @if (running() && !results()) {
            <div class="loading-state">
              @for (_ of [1,2,3,4]; track $index) { <app-skeleton height="80px" /> }
            </div>
          }

          @if (results()) {
            <!-- Strategy tabs -->
            <nav class="strategy-tabs" role="tablist">
              @for (name of resultKeys(); track name) {
                <button class="tab" [class.active]="activeStrategy() === name"
                        (click)="activeStrategy.set(name)" role="tab"
                        [attr.aria-selected]="activeStrategy() === name">
                  <span>{{ meta(name).icon }} {{ meta(name).label }}</span>
                  @if (resultFor(name); as r) {
                    <span class="tab-score">{{ r.totalProjectedScore | number:'1.1-1' }}</span>
                  }
                </button>
              }
            </nav>

            <!-- Diversity insight -->
            @if (results()?.diversity; as div) {
              <div class="insight-banner" [class.warn]="div.lowDiversity">
                <strong>Diversità:</strong> Jaccard {{ div.meanPairwiseJaccard | number:'1.2-2' }}
                @if (div.lowDiversity) { <span class="badge warn">Bassa diversità</span> } @else { <span class="badge ok">OK</span> }
              </div>
            }

            <!-- Monte Carlo summary -->
            @if (multiMcSummary(); as mc) {
              <div class="mc-card">
                <div class="mc-header">
                  <h3>🎲 Monte Carlo</h3>
                  <span class="badge">{{ mc.mode }} · N={{ mc.nSimulations }}</span>
                </div>
                <div class="mc-grid">
                  <div><span class="mc-label">Stability</span> {{ mc.stabilityIndex ?? 0 | number:'1.2-2' }}</div>
                  <div><span class="mc-label">Jaccard medio</span> {{ mc.meanPairwiseJaccard ?? 0 | number:'1.2-2' }}</div>
                  @if (mc.yieldStability?.probAboveThreshold != null) {
                    <div><span class="mc-label">P(yield ≥ soglia)</span> {{ mc.yieldStability!.probAboveThreshold | percent:'1.0-0' }}</div>
                  }
                </div>
                @if (topSelectionFrequency(); as freq) {
                  <details>
                    <summary class="mc-label" style="cursor:pointer">Frequenza selezione (top 12)</summary>
                    <table class="freq-table">
                      @for (row of freq; track row.id) {
                        <tr>
                          <td>{{ row.name }}</td>
                          <td><progress max="1" [value]="row.freq" style="width:60px"></progress> {{ row.freq | percent:'1.0-0' }}</td>
                        </tr>
                      }
                    </table>
                  </details>
                }
              </div>
            }

            @if (activeResult(); as r) {
              <!-- Key metrics -->
              <div class="metrics-grid">
                <div class="metric">
                  <span class="metric-label">Score totale</span>
                  <span class="metric-value">{{ r.totalProjectedScore | number:'1.2-2' }}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Costo nom.</span>
                  <span class="metric-value">{{ r.totalNominalCost }}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Costo eff.</span>
                  <span class="metric-value">{{ r.totalEffectiveCost | number:'1.1-1' }}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Residuo budget</span>
                  <span class="metric-value">{{ r.budgetResidual | number:'1.0-0' }}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Club</span>
                  <span class="metric-value">{{ r.distinctTeamsCount }}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Stato</span>
                  <span class="metric-value" [class.text-green]="r.status === 'Optimal'">{{ r.status === 'Optimal' ? 'Ottimale' : r.status }}</span>
                </div>
                @if (r.winProbability != null) {
                  <div class="metric">
                    <span class="metric-label">Prob. successo budget</span>
                    <span class="metric-value" [style.color]="r.winProbability < 0.4 ? '#EF4444' : r.winProbability < 0.7 ? '#F59E0B' : ''">
                      {{ r.winProbability | percent:'1.0-0' }}
                    </span>
                  </div>
                }
              </div>

              <!-- Formations feasibility -->
              <div class="formation-check">
                <span class="metric-label">Moduli:</span>
                @for (entry of formationEntries(r); track entry[0]) {
                  <span class="form-chip" [class.ok]="entry[1]">{{ entry[0] }} {{ entry[1] ? '✓' : '✗' }}</span>
                }
              </div>

              <!-- Role breakdown -->
              <div class="role-strip">
                @for (role of ['P','D','C','A']; track role) {
                  <span class="role-badge" [style.border-color]="roleColor(role)" [style.color]="roleColor(role)">
                    {{ roleLabel(role) }} {{ r.roleBreakdown[role] || 0 }}
                  </span>
                }
              </div>

              <!-- Squad table -->
              <div class="table-wrap">
                <table class="squad-table">
                  <thead>
                    <tr>
                      <th>Ruolo</th>
                      <th>Giocatore</th>
                      <th class="opt-col">Squadra</th>
                      <th class="num">Costo</th>
                      <th class="num opt-col">Eff.</th>
                      <th class="num">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    @for (p of sortedSquad(r); track p.playerId) {
                      <tr tabindex="0" (click)="selectedPlayer.set(p)" (keydown.enter)="selectedPlayer.set(p)"
                          (keydown.space)="$event.preventDefault(); selectedPlayer.set(p)" role="button"
                          class="clickable-row" [attr.aria-label]="'Dettaglio ' + p.name">
                        <td><span class="role-chip" [style.color]="roleColor(p.role)" [style.border-color]="roleColor(p.role)">{{ roleLabel(p.role) }}</span></td>
                        <td>{{ p.name }}</td>
                        <td class="opt-col">{{ p.realTeam }}</td>
                        <td class="num">{{ p.cost }}</td>
                        <td class="num opt-col muted">{{ p.effectiveCost | number:'1.1-1' }}</td>
                        <td class="num accent">{{ p.projectedScore | number:'1.2-2' }}</td>
                      </tr>
                    }
                  </tbody>
                </table>
              </div>

              <!-- Near-optimal alternatives -->
              @if (r.nearOptimal?.length) {
                <details class="near-opt">
                  <summary>🔁 Alternative near‑optimal ({{ r.nearOptimal.length }})</summary>
                  @for (alt of r.nearOptimal; track $index) {
                    <div class="alt-item">
                      <p>Δ score {{ alt.scoreDelta | number:'1.2-2' }} ({{ alt.scoreDeltaPct | percent:'1.1-1' }}) · esclusi: {{ alt.excludedPlayerIds.join(', ') }}</p>
                      <ul>
                        @for (pl of alt.squad; track pl.playerId) {
                          <li>{{ pl.name }} ({{ pl.role }} · {{ pl.projectedScore | number:'1.1-1' }})</li>
                        }
                      </ul>
                    </div>
                  }
                </details>
              }
            }
          }
        </section>
      </div>
    </div>

    @if (selectedPlayer(); as p) {
      <app-optimizer-player-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
  `,
  styles: [`
    /* ── ROOT & RESET ─────────────────────────────── */
    .optimizer-root {
      display: flex; flex-direction: column;
      height: 100dvh; overflow: hidden;
      background: var(--color-bg, #0e0e10);
      color: var(--color-text-primary, #f8f9fa);
      font-family: var(--font-sans, system-ui, sans-serif);
    }

    /* ── HEADER ───────────────────────────────────── */
    .header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 16px; border-bottom: 1px solid var(--color-border, #2a2a35);
      background: var(--color-surface, #13131a);
      flex-shrink: 0;
    }
    .header-left { display: flex; flex-direction: column; gap: 2px; }
    .header-title { font-size: 1.1rem; font-weight: 700; margin: 0; }
    .header-subtitle { font-size: 0.75rem; color: var(--color-text-secondary, #a1a1aa); margin: 0; }
    .header-run-btn {
      background: var(--color-accent, #6366f1); color: white; border: none;
      padding: 8px 16px; border-radius: 8px; font-weight: 600;
      display: flex; align-items: center; gap: 6px; cursor: pointer;
      transition: opacity 0.15s;
    }
    .header-run-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    .spinner {
      width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── GRID LAYOUT ──────────────────────────────── */
    .main-grid {
      display: flex; flex: 1; overflow: hidden;
    }

    /* ── CONFIG PANEL ─────────────────────────────── */
    .config-panel {
      width: 320px; max-width: 100%; flex-shrink: 0;
      border-right: 1px solid var(--color-border);
      overflow-y: auto; padding: 16px 12px 24px;
      background: var(--color-surface, #13131a);
      display: flex; flex-direction: column; gap: 12px;
    }
    .config-group {
      border: 1px solid var(--color-border); border-radius: 10px;
      background: var(--color-bg, #0e0e10); overflow: hidden;
    }
    .config-group[open] { border-color: var(--color-accent, #6366f1); }
    .config-group-title {
      padding: 10px 12px; font-weight: 600; font-size: 0.85rem;
      cursor: pointer; user-select: none;
      display: flex; align-items: center; gap: 8px;
      background: var(--color-surface-raised, #1a1a22);
    }
    .config-body { padding: 10px 12px; display: flex; flex-direction: column; gap: 10px; }

    .field-label { font-size: 0.75rem; color: var(--color-text-secondary); }
    .field-input {
      background: var(--color-bg); border: 1px solid var(--color-border);
      border-radius: 6px; padding: 8px; font-size: 0.9rem;
      color: var(--color-text-primary); width: 100%;
    }
    .field-textarea { resize: vertical; min-height: 48px; }
    .field-row { display: flex; flex-direction: column; gap: 4px; }
    .preset-desc { font-size: 0.75rem; color: var(--color-text-secondary); margin: 4px 0 0; }
    .muted { opacity: 0.7; }

    .toggle-row {
      display: flex; align-items: center; gap: 8px; cursor: pointer;
      font-size: 0.85rem;
    }
    .warning {
      background: #F59E0B18; border: 1px solid #F59E0B44;
      color: #FBBF24; font-size: 0.75rem; padding: 6px 10px; border-radius: 6px;
    }

    .chip-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; }
    .chip {
      padding: 6px; text-align: center; border: 1px solid var(--color-border);
      border-radius: 8px; cursor: pointer; font-size: 0.8rem; font-weight: 500;
    }
    .chip.active { border-color: var(--color-accent); background: color-mix(in srgb, var(--color-accent) 10%, transparent); }
    .chip input { display: none; }

    .strategy-list { display: flex; flex-direction: column; gap: 6px; }
    .strategy-chip {
      display: flex; align-items: center; gap: 8px; padding: 8px;
      border: 1px solid var(--color-border); border-radius: 8px; cursor: pointer;
    }
    .strategy-chip.active { border-color: var(--color-accent); background: color-mix(in srgb, var(--color-accent) 10%, transparent); }
    .strategy-chip input { display: none; }

    .text-btn { background: none; border: none; color: var(--color-accent); cursor: pointer; font-size: 0.8rem; padding: 4px 0; }
    .weights-panel { display: flex; flex-direction: column; gap: 8px; }
    .weight-row { display: flex; align-items: center; gap: 8px; }
    .weight-row input[type=range] { flex: 1; }
    .weight-value { font-size: 0.8rem; min-width: 36px; }

    .run-btn {
      background: var(--color-accent); color: white; border: none;
      padding: 12px; border-radius: 8px; font-weight: 600;
      display: flex; align-items: center; justify-content: center; gap: 6px;
      cursor: pointer; margin-top: 8px;
    }
    .run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .job-status { font-size: 0.75rem; color: var(--color-text-secondary); margin: 0; }

    /* ── RESULTS PANEL ─────────────────────────────── */
    .results-panel {
      flex: 1; overflow-y: auto; padding: 16px;
      display: flex; flex-direction: column; gap: 16px;
    }
    .empty-state, .loading-state {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center; text-align: center; gap: 12px;
    }
    .empty-icon { font-size: 3rem; }
    .empty-title { font-size: 1.2rem; }
    .empty-desc { color: var(--color-text-secondary); max-width: 320px; }

    .strategy-tabs {
      display: flex; gap: 4px; flex-wrap: wrap; border-bottom: 1px solid var(--color-border);
    }
    .tab {
      display: flex; align-items: center; gap: 6px; padding: 10px 14px;
      background: none; border: none; border-bottom: 2px solid transparent;
      font-size: 0.85rem; color: var(--color-text-secondary); cursor: pointer;
    }
    .tab.active { color: var(--color-accent); border-bottom-color: var(--color-accent); }
    .tab-score { background: var(--color-surface-raised); border-radius: 999px; padding: 2px 8px; font-size: 0.75rem; }

    .insight-banner {
      display: flex; align-items: center; gap: 10px; padding: 10px;
      border-radius: 8px; background: var(--color-surface-raised);
      border: 1px solid var(--color-border);
    }
    .insight-banner.warn { border-color: #F59E0B55; background: #F59E0B12; }
    .badge { padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }
    .badge.warn { background: #F59E0B33; color: #FBBF24; }
    .badge.ok { background: #10B98133; color: #34D399; }

    .mc-card {
      border: 1px solid var(--color-border); border-radius: 10px;
      padding: 12px; background: var(--color-surface);
    }
    .mc-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .mc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 8px; margin-bottom: 8px; }
    .mc-label { font-size: 0.75rem; color: var(--color-text-secondary); }

    .metrics-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(120px,1fr));
      gap: 8px;
    }
    .metric {
      background: var(--color-surface); border: 1px solid var(--color-border);
      border-radius: 8px; padding: 10px;
      display: flex; flex-direction: column; gap: 4px;
    }
    .metric-label { font-size: 0.7rem; color: var(--color-text-secondary); text-transform: uppercase; }
    .metric-value { font-size: 1.1rem; font-weight: 700; }
    .text-green { color: #22C55E; }

    .formation-check {
      display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
      padding: 8px 0;
    }
    .form-chip {
      padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
      background: var(--color-surface-raised); border: 1px solid var(--color-border);
    }
    .form-chip.ok { background: #22C55E18; border-color: #22C55E; color: #22C55E; }

    .role-strip { display: flex; gap: 12px; padding: 4px 0; }
    .role-badge { padding: 2px 8px; border: 1px solid; border-radius: 6px; font-size: 0.75rem; }

    .table-wrap { overflow-x: auto; margin: 0 -16px; padding: 0 16px; }
    .squad-table { width: 100%; min-width: 480px; border-collapse: collapse; font-size: 0.85rem; }
    .squad-table th {
      text-align: left; padding: 8px; border-bottom: 1px solid var(--color-border);
      font-size: 0.7rem; text-transform: uppercase; color: var(--color-text-secondary);
    }
    .squad-table td { padding: 8px; border-bottom: 1px solid var(--color-border); }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .accent { color: var(--color-accent); font-weight: 600; }
    .clickable-row { cursor: pointer; }
    .clickable-row:hover { background: var(--color-surface-raised); }
    .role-chip { display: inline-block; padding: 1px 6px; border: 1px solid; border-radius: 4px; font-size: 0.7rem; }
    .opt-col { display: none; }

    .near-opt { margin-top: 8px; }
    .alt-item { margin: 8px 0; padding-left: 16px; border-left: 2px solid var(--color-border); }

    /* ── RESPONSIVE ────────────────────────────────── */
    @media (max-width: 767px) {
      .main-grid { flex-direction: column; }
      .config-panel {
        width: 100%; border-right: none; border-bottom: 1px solid var(--color-border);
        max-height: 50vh; padding: 12px;
      }
      .header { flex-wrap: wrap; gap: 8px; }
      .header-run-btn { width: 100%; justify-content: center; }
      .opt-col { display: table-cell; } /* show more on mobile? maybe not all, keep some hidden for space */
    }

    @media (min-width: 768px) {
      .opt-col { display: table-cell; }
    }
  `],
})
export class OptimizerComponent {
  // ... (tutto il codice TypeScript esistente rimane identico)
  private readonly optimizerService = inject(OptimizerService);
  private readonly quotationService = inject(QuotationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly allFormations = ALL_FORMATIONS;
  protected readonly OPTIMIZER_LEGENDS = OPTIMIZER_LEGENDS;
  readonly presets: readonly OptimizerPreset[] = OPTIMIZER_PRESETS;
  protected readonly OPTIMIZER_PRESET_NONE = OPTIMIZER_PRESET_NONE;
  readonly selectedPlayer = signal<SquadPlayer | null>(null);
  readonly selectedPresetId = signal<string>(OPTIMIZER_PRESET_NONE);
  readonly activePreset = computed(() => findOptimizerPreset(this.selectedPresetId()));
  readonly availableStrategies = signal<string[]>(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']);

  readonly seasons = signal<number[]>([]);
  readonly seasonsLoading = signal(true);
  readonly seasonStart = signal<number>(2024);
  readonly budget = signal(500);
  readonly numParticipants = signal(8);
  readonly minQtA = signal(1);
  readonly solverTimeoutSeconds = signal(30);

  readonly minDistinctTeams = signal(12);
  readonly maxPlayersPerTeam = signal(4);
  readonly bigTeamsCap = signal(10);
  readonly bigTeamsRaw = signal('Inter, Milan, Juventus, Napoli');
  readonly maxSinglePlayerBudgetShare = signal(0.30);

  readonly mustIncludeRaw = signal('');
  readonly excludeRaw = signal('');

  readonly selectedFormations = signal(new Set(ALL_FORMATIONS.map(f => f.label)));
  readonly preferredFormationLabel = signal<string>('');

  readonly ruleset = signal<'CLASSIC' | 'MANTRA'>('CLASSIC');

  readonly inflationPercentileThreshold = signal(0.7);
  readonly maxInflationMultiplier = signal(1.6);
  readonly baseInflationRate = signal(0.05);
  readonly baselineParticipants = signal(8);
  readonly teamStrengthMultiplier = signal(0.0);

  readonly riskAversion = signal(0.0);
  readonly monteCarloEnabled = signal(false);
  readonly monteCarloMode = signal<'mean_std' | 'saa_frequency'>('saa_frequency');
  readonly nSimulations = signal(10);
  readonly riskLambda = signal(0.5);
  readonly nearOptimalEnabled = signal(false);

  readonly varBlend = signal(0.0);
  readonly esvWeight = signal(0.0);
  readonly valuationMode = signal<'PER_MATCH_RATING' | 'SEASON_VALUE'>('PER_MATCH_RATING');
  readonly minStartProbability = signal<number | null>(null);
  readonly replacementMethod = signal<'percentile' | 'roster_depth'>('percentile');

  readonly selectedStrategies = signal(new Set(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']));
  readonly strategyProfilesMap = signal<Record<string, StrategyProfile>>({});
  readonly showCustomWeights = signal(false);
  readonly customWeights = signal<Record<string, number>>({ P: 1, D: 1, C: 1, A: 1 });

  readonly singleStrategySelected = computed(() => this.selectedStrategies().size === 1);

  readonly running = signal(false);
  readonly jobStatus = signal<string | null>(null);
  readonly jobId = signal<string | null>(null);
  readonly usedAsyncJob = signal(false);

  readonly error = signal<string | null>(null);
  readonly results = signal<MultiStrategyResult | null>(null);
  readonly activeStrategy = signal<string>('');

  readonly resultKeys = computed(() => Object.keys(this.results()?.results ?? {}));
  readonly multiMcSummary = computed(() => {
    const res = this.results();
    if (!res) return null;
    return res.monteCarloSummary
      ?? res.results[this.activeStrategy()]?.monteCarloSummary
      ?? null;
  });
  readonly playerNameById = computed(() => {
    const map = new Map<string, string>();
    const res = this.results();
    if (!res) return map;
    for (const r of Object.values(res.results)) {
      for (const pl of r.squad ?? []) {
        if (pl.playerId) map.set(pl.playerId, pl.name || pl.playerId);
      }
      for (const alt of r.nearOptimal ?? []) {
        for (const pl of alt.squad ?? []) {
          if (pl.playerId) map.set(pl.playerId, pl.name || pl.playerId);
        }
      }
    }
    return map;
  });

  readonly topSelectionFrequency = computed(() => {
    const freq = this.multiMcSummary()?.selectionFrequency;
    if (!freq) return [] as { id: string; name: string; freq: number }[];
    const names = this.playerNameById();
    return Object.entries(freq)
      .map(([id, f]) => ({
        id,
        name: names.get(id) ?? id,
        freq: f as number,
      }))
      .sort((a, b) => b.freq - a.freq)
      .slice(0, 12);
  });

  readonly activeResult = computed((): OptimizationResult | null =>
    this.results()?.results[this.activeStrategy()] ?? null,
  );
  readonly canRun = computed(() =>
    this.selectedStrategies().size > 0 &&
    this.selectedFormations().size > 0 &&
    this.seasons().length > 0,
  );

  constructor() {
    this.quotationService.getSeasons().subscribe({
      next: s => {
        const sorted = [...s].sort((a, b) => b - a);
        this.seasons.set(sorted);
        if (sorted.length) this.seasonStart.set(sorted[0]);
        this.seasonsLoading.set(false);
      },
      error: () => {
        this.seasons.set([2024, 2023, 2022]);
        this.seasonsLoading.set(false);
      },
    });

    this.optimizerService.getStrategies().subscribe({
      next: res => {
        const names = res.strategies.map((s: StrategyProfile) => s.name);
        this.availableStrategies.set(names);
        this.selectedStrategies.set(new Set(names));
        const map: Record<string, StrategyProfile> = {};
        res.strategies.forEach((s: StrategyProfile) => { map[s.name] = s; });
        this.strategyProfilesMap.set(map);
      },
      error: () => { /* keep fallback */ },
    });
  }

  onPresetChange(presetId: string): void {
    this.selectedPresetId.set(presetId ?? OPTIMIZER_PRESET_NONE);
    const preset = findOptimizerPreset(presetId);
    if (preset) {
      this.applyPreset(preset);
    }
  }

  applyPreset(preset: OptimizerPreset): void {
    const req = preset.request;

    if (req.budget != null) this.budget.set(req.budget);
    if (req.numParticipants != null) this.numParticipants.set(req.numParticipants);
    if (req.minQtA != null) this.minQtA.set(req.minQtA);
    if (req.solverTimeoutSeconds != null) this.solverTimeoutSeconds.set(req.solverTimeoutSeconds);
    if (req.minDistinctTeams != null) this.minDistinctTeams.set(req.minDistinctTeams);
    if (req.maxPlayersPerTeam != null) this.maxPlayersPerTeam.set(req.maxPlayersPerTeam);
    if (req.bigTeamsCap != null) this.bigTeamsCap.set(req.bigTeamsCap);
    if (req.maxSinglePlayerBudgetShare != null) {
      this.maxSinglePlayerBudgetShare.set(req.maxSinglePlayerBudgetShare);
    }
    if (req.bigTeams?.length) {
      this.bigTeamsRaw.set(req.bigTeams.join(', '));
    }
    if (req.ruleset === 'CLASSIC' || req.ruleset === 'MANTRA') {
      this.ruleset.set(req.ruleset);
    }
    if (req.riskAversion != null) this.riskAversion.set(req.riskAversion);
    if (req.monteCarlo != null) {
      this.monteCarloEnabled.set(!!req.monteCarlo.enabled);
      if (req.monteCarlo.mode) this.monteCarloMode.set(req.monteCarlo.mode);
      if (req.monteCarlo.nSimulations != null) this.nSimulations.set(req.monteCarlo.nSimulations);
      if (req.monteCarlo.riskLambda != null) this.riskLambda.set(req.monteCarlo.riskLambda);
    } else {
      this.monteCarloEnabled.set(false);
    }
    if (req.nearOptimal != null) {
      this.nearOptimalEnabled.set(!!req.nearOptimal.enabled);
    }

    if (req.varBlend != null) this.varBlend.set(req.varBlend);
    if (req.esvWeight != null) this.esvWeight.set(req.esvWeight);
    if (req.valuationMode === 'PER_MATCH_RATING' || req.valuationMode === 'SEASON_VALUE') {
      this.valuationMode.set(req.valuationMode);
    }
    if (req.replacementMethod === 'percentile' || req.replacementMethod === 'roster_depth') {
      this.replacementMethod.set(req.replacementMethod);
    }
    if (req.minStartProbability === null) {
      this.minStartProbability.set(null);
    } else if (typeof req.minStartProbability === 'number') {
      this.minStartProbability.set(req.minStartProbability);
    }

    const infl = req.inflationConfig;
    if (infl) {
      if (infl.inflationPercentileThreshold != null) {
        this.inflationPercentileThreshold.set(infl.inflationPercentileThreshold);
      }
      if (infl.maxInflationMultiplier != null) {
        this.maxInflationMultiplier.set(infl.maxInflationMultiplier);
      }
      if (infl.baseInflationRate != null) {
        this.baseInflationRate.set(infl.baseInflationRate);
      }
      if (infl.baselineParticipants != null) {
        this.baselineParticipants.set(infl.baselineParticipants);
      }
      if (infl.teamStrengthMultiplier != null) {
        this.teamStrengthMultiplier.set(infl.teamStrengthMultiplier);
      }
    }

    if (req.formations?.length) {
      this.selectedFormations.set(new Set(req.formations.map(f => f.label)));
    }

    if (req.preferredFormation?.label) {
      this.preferredFormationLabel.set(req.preferredFormation.label);
    } else if (req.preferredFormation === null) {
      this.preferredFormationLabel.set('');
    }

    if (req.customStrategies?.length) {
      const names = req.customStrategies.map(s => s.name);
      this.selectedStrategies.set(new Set(names));
      const primary = req.customStrategies[0];
      this.customWeights.set({ P: 1, D: 1, C: 1, A: 1, ...primary.roleWeight });
      this.showCustomWeights.set(true);
      this.strategyProfilesMap.update(map => {
        const next = { ...map };
        for (const s of req.customStrategies!) {
          next[s.name] = s;
        }
        return next;
      });
    } else if (req.strategyNames?.length) {
      this.selectedStrategies.set(new Set(req.strategyNames));
      this.showCustomWeights.set(false);
    }
  }

  toggleStrategy(name: string): void {
    this.selectedPresetId.set(OPTIMIZER_PRESET_NONE);
    this.selectedStrategies.update(s => {
      const n = new Set(s); n.has(name) ? n.delete(name) : n.add(name); return n;
    });
    const sel = this.selectedStrategies();
    if (sel.size === 1) {
      const profile = this.strategyProfilesMap()[[...sel][0]];
      if (profile) this.customWeights.set({ ...profile.roleWeight });
    }
    if (sel.size !== 1) this.showCustomWeights.set(false);
  }

  toggleFormation(label: string): void {
    this.selectedFormations.update(s => {
      const n = new Set(s); n.has(label) ? n.delete(label) : n.add(label); return n;
    });
  }

  run(): void {
    this.running.set(true);
    this.error.set(null);
    this.jobStatus.set(null);
    this.jobId.set(null);
    this.usedAsyncJob.set(false);

    const req = this._buildRequest();
    const useAsync =
      !!req.monteCarlo?.enabled &&
      (req.monteCarlo.nSimulations ?? 0) > ASYNC_MC_THRESHOLD;

    if (useAsync) {
      this.usedAsyncJob.set(true);
      this._runAsyncJob(req);
      return;
    }

    this.optimizerService.runMulti(req).subscribe({
      next: res => {
        this.results.set(res);
        this.activeStrategy.set(Object.keys(res.results)[0] ?? '');
        this.running.set(false);
      },
      error: err => {
        this.error.set(err.error?.detail ?? err.message ?? 'Unknown error');
        this.running.set(false);
      },
    });
  }

  private _runAsyncJob(req: OptimizationRequest): void {
    const strategyName = this._resolveStrategyName(req);

    this.jobStatus.set('queued');
    this.optimizerService
      .createJob(req, strategyName)
      .pipe(
        switchMap(created => {
          this.jobId.set(created.jobId);
          this.jobStatus.set(created.status || 'queued');
          return interval(JOB_POLL_MS).pipe(
            startWith(0),
            take(JOB_MAX_POLLS),
            switchMap(() => this.optimizerService.pollJobStatus(created.jobId)),
            takeWhile(
              job => job.status === 'queued' || job.status === 'running',
              true,
            ),
          );
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: job => {
          this.jobStatus.set(job.status);
          if (job.status === 'completed' && job.result) {
            const name = job.result.strategyName || strategyName;
            const multi: MultiStrategyResult = {
              results: { [name]: job.result },
              monteCarloSummary:
                job.monteCarloSummary ?? job.result.monteCarloSummary ?? null,
              diversity: null,
            };
            this.results.set(multi);
            this.activeStrategy.set(name);
            this.running.set(false);
          } else if (job.status === 'failed') {
            this.error.set(job.error || 'Job Monte Carlo fallito');
            this.running.set(false);
          }
        },
        error: err => {
          this.error.set(err.error?.detail ?? err.message ?? 'Job async error');
          this.running.set(false);
          this.jobStatus.set('failed');
        },
        complete: () => {
          if (this.running()) {
            this.error.set(
              'Timeout polling job async: aumenta N gradualmente o riprova più tardi.',
            );
            this.running.set(false);
            this.jobStatus.set('timeout');
          }
        },
      });
  }

  private _buildRequest(): OptimizationRequest {
    const bigTeams = this.bigTeamsRaw()
      .split(',').map(t => t.trim()).filter(Boolean);

    const formations = ALL_FORMATIONS.filter(f => this.selectedFormations().has(f.label));
    const preferredFormation =
      formations.find(f => f.label === this.preferredFormationLabel()) ?? null;

    const mustInclude = this.mustIncludeRaw()
      .split(/[\n,]+/).map(s => s.trim()).filter(Boolean);
    const exclude = this.excludeRaw()
      .split(/[\n,]+/).map(s => s.trim()).filter(Boolean);

    return {
      seasonStart: this.seasonStart(),
      budget: this.budget(),
      numParticipants: this.numParticipants(),
      minQtA: this.minQtA(),
      minDistinctTeams: this.minDistinctTeams(),
      maxPlayersPerTeam: this.maxPlayersPerTeam(),
      solverTimeoutSeconds: this.solverTimeoutSeconds(),
      bigTeamsCap: this.bigTeamsCap(),
      bigTeams,
      formations,
      inflationConfig: {
        inflationPercentileThreshold: this.inflationPercentileThreshold(),
        maxInflationMultiplier: this.maxInflationMultiplier(),
        baseInflationRate: this.baseInflationRate(),
        baselineParticipants: this.baselineParticipants(),
        teamStrengthMultiplier: this.teamStrengthMultiplier(),
      },
      maxSinglePlayerBudgetShare: this.maxSinglePlayerBudgetShare(),
      mustInclude: mustInclude.length ? mustInclude : undefined,
      exclude: exclude.length ? exclude : undefined,
      ruleset: this.ruleset(),
      preferredFormation,
      riskAversion: this.riskAversion(),
      monteCarlo: this.monteCarloEnabled()
        ? {
            enabled: true,
            nSimulations: this.nSimulations(),
            mode: this.monteCarloMode(),
            riskLambda: this.riskLambda(),
            randomSeed: 42,
          }
        : undefined,
      nearOptimal: this.nearOptimalEnabled()
        ? { enabled: true, nAlternatives: 3, excludeTopM: 2, maxScoreDropPct: 0.15 }
        : undefined,
      varBlend: this.varBlend(),
      esvWeight: this.esvWeight(),
      valuationMode: this.valuationMode(),
      minStartProbability: this.minStartProbability(),
      replacementMethod: this.replacementMethod(),
      strategyNames: this.showCustomWeights() ? null : [...this.selectedStrategies()],
      customStrategies: this.showCustomWeights() ? this._buildCustomStrategies() : null,
    };
  }

  setCustomWeight(role: string, value: number): void {
    this.customWeights.update(w => ({ ...w, [role]: +value }));
  }

  private _resolveStrategyName(req: OptimizationRequest): string {
      if (req.customStrategies?.length) {
        return req.customStrategies[0].name || 'BALANCED';
      }
      if (req.strategyNames?.length) {
        return req.strategyNames[0];
      }
    const selected = [...this.selectedStrategies()];
    return selected[0] ?? 'BALANCED';
  }

  private _buildCustomStrategies(): StrategyProfile[] {
    const [name] = [...this.selectedStrategies()];
    const base = this.strategyProfilesMap()[name];
    return [{
      name,
      roleWeight: { ...this.customWeights() },
      minBudgetShareByRoles: base?.minBudgetShareByRoles ?? null,
      maxTopTierPlayers: base?.maxTopTierPlayers ?? null,
      topTierCostThreshold: base?.topTierCostThreshold ?? null,
    }];
  }

  resultFor(name: string): OptimizationResult | null {
    return this.results()?.results[name] ?? null;
  }

  sortedSquad(r: OptimizationResult): SquadPlayer[] {
    const order = ['P', 'D', 'C', 'A'];
    return [...r.squad].sort((a, b) =>
      order.indexOf(a.role) - order.indexOf(b.role) || b.projectedScore - a.projectedScore,
    );
  }

  formationEntries(r: OptimizationResult): [string, boolean][] {
    return Object.entries(r.formationFeasibility);
  }

  meta(name: string) {
    return STRATEGY_META[name] ?? { label: name, icon: '📋' };
  }

  roleColor(role: string): string { return ROLE_COLORS[role] ?? 'var(--color-text-secondary)'; }
  roleLabel(role: string): string { return ROLE_LABELS[role] ?? role; }
}