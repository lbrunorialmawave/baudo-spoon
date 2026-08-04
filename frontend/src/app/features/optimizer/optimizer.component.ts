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
    <div class="optimizer-page">

      <header class="page-header">
        <div>
          <h1 class="page-title">Ottimizzatore Rosa</h1>
          <p class="page-subtitle">ILP + robustezza opzionale Monte Carlo: score, vincoli, stability e alternative near-optimal</p>
        </div>
      </header>

      <div class="optimizer-body">

        <!-- ── Config panel ──────────────────────────────── -->
        <aside class="config-panel card">

          <!-- PRESETS -->
          <p class="section-divider">Profilo strategico</p>

          <div class="field-group">
            <label class="field-label" for="opt-preset">Preset strategia (precompila leve obiettivo e vincoli)</label>
            <select
              id="opt-preset"
              class="field-input"
              [ngModel]="selectedPresetId()"
              (ngModelChange)="onPresetChange($event)"
              [attr.aria-describedby]="'legend-preset'"
            >
              <option [ngValue]="OPTIMIZER_PRESET_NONE">Personalizzato (nessun preset)</option>
              @for (p of presets; track p.id) {
                <option [ngValue]="p.id">{{ p.labelIt }} — {{ p.name }}</option>
              }
            </select>
            @if (activePreset(); as preset) {
              <p class="preset-description" id="legend-preset">{{ preset.description }}</p>
            } @else {
              <p class="preset-description muted" id="legend-preset">
                Scegli un profilo per precompilare vincoli, rischio, inflazione e strategie.
                Stagione, include/exclude restano sotto il tuo controllo.
              </p>
            }
          </div>

          <!-- BASIC -->
          <p class="section-divider">Pool e budget</p>

          <div class="field-group">
            <label class="field-label" for="opt-seasonStart">Stagione del pool (listini + predizioni ML)</label>
            @if (seasonsLoading()) {
              <app-skeleton height="36px" />
            } @else {
              <select id="opt-seasonStart" class="field-input" [(ngModel)]="seasonStart"
                      [attr.aria-describedby]="'legend-seasonStart'">
                @for (s of seasons(); track s) {
                  <option [value]="s">{{ s }}/{{ s + 1 }}</option>
                }
              </select>
            }
            <app-field-legend
              fieldId="legend-seasonStart"
              [description]="OPTIMIZER_LEGENDS['seasonStart'].description"
              [examples]="OPTIMIZER_LEGENDS['seasonStart'].examples" />
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-budget">Budget rosa (tetto costi effettivi) <span class="field-hint">cr.</span></label>
              <input id="opt-budget" class="field-input" type="number" min="200" max="1000" step="25"
                     [(ngModel)]="budget"
                     [attr.aria-describedby]="'legend-budget'" />
              <app-field-legend
                fieldId="legend-budget"
                [description]="OPTIMIZER_LEGENDS['budget'].description"
                [examples]="OPTIMIZER_LEGENDS['budget'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-numParticipants">Partecipanti lega (spinge l'inflazione dei costi)</label>
              <input id="opt-numParticipants" class="field-input" type="number" min="4" max="16" step="1"
                     [(ngModel)]="numParticipants"
                     [attr.aria-describedby]="'legend-numParticipants'" />
              <app-field-legend
                fieldId="legend-numParticipants"
                [description]="OPTIMIZER_LEGENDS['numParticipants'].description"
                [examples]="OPTIMIZER_LEGENDS['numParticipants'].examples" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-minQtA">Listino minimo per entrare nel pool <span class="field-hint">qt_a ≥</span></label>
              <input id="opt-minQtA" class="field-input" type="number" min="0" max="10" step="1"
                     [(ngModel)]="minQtA"
                     [attr.aria-describedby]="'legend-minQtA'" />
              <app-field-legend
                fieldId="legend-minQtA"
                [description]="OPTIMIZER_LEGENDS['minQtA'].description"
                [examples]="OPTIMIZER_LEGENDS['minQtA'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-solverTimeout">Timeout solver ILP <span class="field-hint">secondi, non cambia l'obiettivo</span></label>
              <input id="opt-solverTimeout" class="field-input" type="number" min="5" max="300" step="5"
                     [(ngModel)]="solverTimeoutSeconds"
                     [attr.aria-describedby]="'legend-solverTimeoutSeconds'" />
              <app-field-legend
                fieldId="legend-solverTimeoutSeconds"
                [description]="OPTIMIZER_LEGENDS['solverTimeoutSeconds'].description"
                [examples]="OPTIMIZER_LEGENDS['solverTimeoutSeconds'].examples" />
            </div>
          </div>

          <!-- SQUAD CONSTRAINTS -->
          <p class="section-divider">Vincoli hard sulla rosa</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-minDistinctTeams">Min. club distinti in rosa <span class="field-hint">vincolo hard</span></label>
              <input id="opt-minDistinctTeams" class="field-input" type="number" min="1" max="25" step="1"
                     [(ngModel)]="minDistinctTeams"
                     [attr.aria-describedby]="'legend-minDistinctTeams'" />
              <app-field-legend
                fieldId="legend-minDistinctTeams"
                [description]="OPTIMIZER_LEGENDS['minDistinctTeams'].description"
                [examples]="OPTIMIZER_LEGENDS['minDistinctTeams'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-maxPlayersPerTeam">Max giocatori dallo stesso club <span class="field-hint">vincolo hard</span></label>
              <input id="opt-maxPlayersPerTeam" class="field-input" type="number" min="1" max="10" step="1"
                     [(ngModel)]="maxPlayersPerTeam"
                     [attr.aria-describedby]="'legend-maxPlayersPerTeam'" />
              <app-field-legend
                fieldId="legend-maxPlayersPerTeam"
                [description]="OPTIMIZER_LEGENDS['maxPlayersPerTeam'].description"
                [examples]="OPTIMIZER_LEGENDS['maxPlayersPerTeam'].examples" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-bigTeamsCap">Tetto giocatori dalle big team <span class="field-hint">vincolo hard aggregato</span></label>
              <input id="opt-bigTeamsCap" class="field-input" type="number" min="0" max="25" step="1"
                     [(ngModel)]="bigTeamsCap"
                     [attr.aria-describedby]="'legend-bigTeamsCap'" />
              <app-field-legend
                fieldId="legend-bigTeamsCap"
                [description]="OPTIMIZER_LEGENDS['bigTeamsCap'].description"
                [examples]="OPTIMIZER_LEGENDS['bigTeamsCap'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-maxShare">Max costo effettivo su un giocatore <span class="field-hint">frazione del budget</span></label>
              <input id="opt-maxShare" class="field-input" type="number" min="0.05" max="1" step="0.05"
                     [(ngModel)]="maxSinglePlayerBudgetShare"
                     [attr.aria-describedby]="'legend-maxSinglePlayerBudgetShare'" />
              <app-field-legend
                fieldId="legend-maxSinglePlayerBudgetShare"
                [description]="OPTIMIZER_LEGENDS['maxSinglePlayerBudgetShare'].description"
                [examples]="OPTIMIZER_LEGENDS['maxSinglePlayerBudgetShare'].examples" />
            </div>
          </div>

          <div class="field-group">
            <label class="field-label" for="opt-bigTeamsRaw">Club conteggiati come big team <span class="field-hint">nomi real_team, separati da virgola</span></label>
            <textarea id="opt-bigTeamsRaw" class="field-input field-textarea" rows="2"
                      [(ngModel)]="bigTeamsRaw"
                      placeholder="Inter, Milan, Juventus, Napoli"
                      [attr.aria-describedby]="'legend-bigTeams'"></textarea>
            <app-field-legend
              fieldId="legend-bigTeams"
              [description]="OPTIMIZER_LEGENDS['bigTeams'].description"
              [examples]="OPTIMIZER_LEGENDS['bigTeams'].examples" />
          </div>

          <!-- PLAYER FILTERS -->
          <p class="section-divider">Include / exclude (vincoli hard)</p>

          <div class="field-group">
            <label class="field-label" for="opt-mustInclude">Must-include <span class="field-hint">player_id obbligatori, vincolo hard</span></label>
            <textarea id="opt-mustInclude" class="field-input field-textarea" rows="2"
                      [(ngModel)]="mustIncludeRaw"
                      placeholder="fm-12345, fm-67890"
                      [attr.aria-describedby]="'legend-mustInclude'"></textarea>
            <app-field-legend
              fieldId="legend-mustInclude"
              [description]="OPTIMIZER_LEGENDS['mustInclude'].description"
              [examples]="OPTIMIZER_LEGENDS['mustInclude'].examples" />
          </div>

          <div class="field-group">
            <label class="field-label" for="opt-exclude">Exclude <span class="field-hint">player_id fuori dal pool</span></label>
            <textarea id="opt-exclude" class="field-input field-textarea" rows="2"
                      [(ngModel)]="excludeRaw"
                      placeholder="fm-12345, fm-67890"
                      [attr.aria-describedby]="'legend-exclude'"></textarea>
            <app-field-legend
              fieldId="legend-exclude"
              [description]="OPTIMIZER_LEGENDS['exclude'].description"
              [examples]="OPTIMIZER_LEGENDS['exclude'].examples" />
          </div>

          <!-- RULESET & RISK -->
          <p class="section-divider">Ruleset e funzione obiettivo</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-ruleset">Ruleset quote/ruoli <span class="field-hint">CLASSIC 4 ruoli · MANTRA 12</span></label>
              <select id="opt-ruleset" class="field-input" [(ngModel)]="ruleset"
                      [attr.aria-describedby]="'legend-ruleset'">
                <option value="CLASSIC">CLASSIC — quote P3/D8/C8/A6</option>
                <option value="MANTRA">MANTRA — 12 ruoli multi-slot</option>
              </select>
              <app-field-legend
                fieldId="legend-ruleset"
                [description]="OPTIMIZER_LEGENDS['ruleset'].description"
                [examples]="OPTIMIZER_LEGENDS['ruleset'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-riskAversion">Risk aversion <span class="field-hint">penalità × prediction_std in obiettivo</span></label>
              <input id="opt-riskAversion" class="field-input" type="number" min="0" max="5" step="0.1"
                     [(ngModel)]="riskAversion"
                     [attr.aria-describedby]="'legend-riskAversion'" />
              <app-field-legend
                fieldId="legend-riskAversion"
                [description]="OPTIMIZER_LEGENDS['riskAversion'].description"
                [examples]="OPTIMIZER_LEGENDS['riskAversion'].examples" />
            </div>

          <!-- MONTE CARLO ROBUSTNESS -->
          <p class="section-divider">Robustezza Monte Carlo</p>

          <div class="field-group field-group--toggle">
            <label class="field-label" for="opt-mc-enabled">
              <input id="opt-mc-enabled" type="checkbox"
                     [ngModel]="monteCarloEnabled()"
                     (ngModelChange)="monteCarloEnabled.set($event)"
                     [attr.aria-describedby]="'legend-mc-enabled'" />
              Abilita Monte Carlo <span class="field-hint">default off = ILP deterministico</span>
            </label>
            <app-field-legend
              fieldId="legend-mc-enabled"
              [description]="OPTIMIZER_LEGENDS['monteCarloEnabled'].description"
              [examples]="OPTIMIZER_LEGENDS['monteCarloEnabled'].examples" />
          </div>

          @if (monteCarloEnabled()) {
            <div class="field-row">
              <div class="field-group">
                <label class="field-label" for="opt-mc-mode">Mode</label>
                <select id="opt-mc-mode" class="field-input"
                        [ngModel]="monteCarloMode()"
                        (ngModelChange)="monteCarloMode.set($event)"
                        [attr.aria-describedby]="'legend-mc-mode'">
                  <option value="saa_frequency">saa_frequency — frequenza scenari</option>
                  <option value="mean_std">mean_std — mean − λ·std</option>
                </select>
                <app-field-legend
                  fieldId="legend-mc-mode"
                  [description]="OPTIMIZER_LEGENDS['monteCarloMode'].description"
                  [examples]="OPTIMIZER_LEGENDS['monteCarloMode'].examples" />
              </div>
              <div class="field-group">
                <label class="field-label" for="opt-mc-n">N simulazioni</label>
                <input id="opt-mc-n" class="field-input" type="number" min="1" max="200" step="1"
                       [ngModel]="nSimulations()"
                       (ngModelChange)="nSimulations.set(+$event)"
                       [attr.aria-describedby]="'legend-mc-n'" />
                <app-field-legend
                  fieldId="legend-mc-n"
                  [description]="OPTIMIZER_LEGENDS['nSimulations'].description"
                  [examples]="OPTIMIZER_LEGENDS['nSimulations'].examples" />
              </div>
            </div>
            @if (monteCarloMode() === 'mean_std') {
              <div class="field-group">
                <label class="field-label" for="opt-mc-lambda">Risk λ (mean_std)</label>
                <input id="opt-mc-lambda" class="field-input" type="number" min="0" max="3" step="0.1"
                       [ngModel]="riskLambda()"
                       (ngModelChange)="riskLambda.set(+$event)" />
              </div>
            }
            @if (riskAversion() > 0 && monteCarloEnabled()) {
              <p class="field-warning" role="status">
                Attenzione: riskAversion={{ riskAversion() }} e Monte Carlo sono entrambi attivi.
                Preferisci uno dei due o tieni riskAversion basso (≤0.3) per evitare doppia penalizzazione.
              </p>
            }
          }

          <div class="field-group field-group--toggle">
            <label class="field-label" for="opt-near-opt">
              <input id="opt-near-opt" type="checkbox"
                     [ngModel]="nearOptimalEnabled()"
                     (ngModelChange)="nearOptimalEnabled.set($event)"
                     [attr.aria-describedby]="'legend-near-opt'" />
              Alternative near-optimal <span class="field-hint">esclude top scorer e ri-ottimizza</span>
            </label>
            <app-field-legend
              fieldId="legend-near-opt"
              [description]="OPTIMIZER_LEGENDS['nearOptimal'].description"
              [examples]="OPTIMIZER_LEGENDS['nearOptimal'].examples" />
          </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-varBlend">VAR blend <span class="field-hint">(1−w)×metric + w×var_score</span></label>
              <input id="opt-varBlend" class="field-input" type="number" min="0" max="1" step="0.1"
                     [(ngModel)]="varBlend"
                     [attr.aria-describedby]="'legend-varBlend'" />
              <app-field-legend
                fieldId="legend-varBlend"
                [description]="OPTIMIZER_LEGENDS['varBlend'].description"
                [examples]="OPTIMIZER_LEGENDS['varBlend'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-esvWeight">ESV weight <span class="field-hint">+ w×surplus value (affare)</span></label>
              <input id="opt-esvWeight" class="field-input" type="number" min="0" max="5" step="0.1"
                     [(ngModel)]="esvWeight"
                     [attr.aria-describedby]="'legend-esvWeight'" />
              <app-field-legend
                fieldId="legend-esvWeight"
                [description]="OPTIMIZER_LEGENDS['esvWeight'].description"
                [examples]="OPTIMIZER_LEGENDS['esvWeight'].examples" />
            </div>
          </div>

          <!-- VAR/ESV ADVANCED -->
          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-valuationMode">Metrica base in obiettivo</label>
              <select id="opt-valuationMode" class="field-input" [(ngModel)]="valuationMode"
                      [attr.aria-describedby]="'legend-valuationMode'">
                <option value="PER_MATCH_RATING">projected_score / partita (default)</option>
                <option value="SEASON_VALUE">season_value (rating × presenze)</option>
              </select>
              <app-field-legend
                fieldId="legend-valuationMode"
                [description]="OPTIMIZER_LEGENDS['valuationMode'].description"
                [examples]="OPTIMIZER_LEGENDS['valuationMode'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-replacementMethod">Replacement level (per VAR/ESV)</label>
              <select id="opt-replacementMethod" class="field-input" [(ngModel)]="replacementMethod"
                      [attr.aria-describedby]="'legend-replacementMethod'">
                <option value="percentile">Percentile basso per ruolo (default)</option>
                <option value="roster_depth">Roster depth (quota × partecipanti)</option>
              </select>
              <app-field-legend
                fieldId="legend-replacementMethod"
                [description]="OPTIMIZER_LEGENDS['replacementMethod'].description"
                [examples]="OPTIMIZER_LEGENDS['replacementMethod'].examples" />
            </div>
          </div>

          <div class="field-group">
            <label class="field-label" for="opt-minStartProb">Filtro start_probability minima <span class="field-hint">pre-ILP, vuoto = off</span></label>
            <input id="opt-minStartProb" class="field-input" type="number" min="0" max="1" step="0.05"
                   [ngModel]="minStartProbability()"
                   (ngModelChange)="minStartProbability.set($event === '' ? null : +$event)"
                   [attr.aria-describedby]="'legend-minStartProbability'" />
            <app-field-legend
              fieldId="legend-minStartProbability"
              [description]="OPTIMIZER_LEGENDS['minStartProbability'].description"
              [examples]="OPTIMIZER_LEGENDS['minStartProbability'].examples" />
          </div>

          <!-- FORMATIONS -->
          <p class="section-divider">Moduli (check post-hoc e vincolo hard)</p>

          <div class="check-grid" role="group" aria-label="Moduli tattici ammessi">
            @for (f of allFormations; track f.label) {
              <label class="check-chip" [class.active]="selectedFormations().has(f.label)">
                <input type="checkbox" [checked]="selectedFormations().has(f.label)"
                       (change)="toggleFormation(f.label)" />
                {{ f.label }}
              </label>
            }
          </div>
          <app-field-legend
            fieldId="legend-formations"
            [description]="OPTIMIZER_LEGENDS['formations'].description"
            [examples]="OPTIMIZER_LEGENDS['formations'].examples" />

          <div class="field-group">
            <label class="field-label" for="opt-preferredFormation">Modulo imposto al solver <span class="field-hint">vincolo hard; le altre solo check</span></label>
            <select id="opt-preferredFormation" class="field-input" [(ngModel)]="preferredFormationLabel"
                    [attr.aria-describedby]="'legend-preferredFormation'">
              <option value="">Nessuna (nessun vincolo)</option>
              @for (f of allFormations; track f.label) {
                <option [value]="f.label">{{ f.label }}</option>
              }
            </select>
            <app-field-legend
              fieldId="legend-preferredFormation"
              [description]="OPTIMIZER_LEGENDS['preferredFormation'].description"
              [examples]="OPTIMIZER_LEGENDS['preferredFormation'].examples" />
          </div>

          <!-- INFLATION MODEL -->
          <p class="section-divider">Costo effettivo (inflazione listino)</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-inflationPercentile">Soglia percentile: sotto = costo = listino</label>
              <input id="opt-inflationPercentile" class="field-input" type="number" min="0" max="1" step="0.05"
                     [(ngModel)]="inflationPercentileThreshold"
                     [attr.aria-describedby]="'legend-inflationPercentileThreshold'" />
              <app-field-legend
                fieldId="legend-inflationPercentileThreshold"
                [description]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].description"
                [examples]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-maxInflation">Cap moltiplicatore costo effettivo / listino</label>
              <input id="opt-maxInflation" class="field-input" type="number" min="1" max="5" step="0.1"
                     [(ngModel)]="maxInflationMultiplier"
                     [attr.aria-describedby]="'legend-maxInflationMultiplier'" />
              <app-field-legend
                fieldId="legend-maxInflationMultiplier"
                [description]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].description"
                [examples]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].examples" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-baseRate">Tasso base inflazione (partecipanti extra)</label>
              <input id="opt-baseRate" class="field-input" type="number" min="0" max="1" step="0.01"
                     [(ngModel)]="baseInflationRate"
                     [attr.aria-describedby]="'legend-baseInflationRate'" />
              <app-field-legend
                fieldId="legend-baseInflationRate"
                [description]="OPTIMIZER_LEGENDS['baseInflationRate'].description"
                [examples]="OPTIMIZER_LEGENDS['baseInflationRate'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-baselinePart">Baseline partecipanti (oltre → extra inflazione)</label>
              <input id="opt-baselinePart" class="field-input" type="number" min="2" max="20" step="1"
                     [(ngModel)]="baselineParticipants"
                     [attr.aria-describedby]="'legend-baselineParticipants'" />
              <app-field-legend
                fieldId="legend-baselineParticipants"
                [description]="OPTIMIZER_LEGENDS['baselineParticipants'].description"
                [examples]="OPTIMIZER_LEGENDS['baselineParticipants'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-teamStrengthMul">Peso Elo club sul costo effettivo <span class="field-hint">0 = off</span></label>
              <input id="opt-teamStrengthMul" class="field-input" type="number" min="0" max="2" step="0.05"
                     [(ngModel)]="teamStrengthMultiplier"
                     [attr.aria-describedby]="'legend-teamStrengthMultiplier'" />
              <app-field-legend
                fieldId="legend-teamStrengthMultiplier"
                [description]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].description"
                [examples]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].examples" />
            </div>
          </div>

          <!-- STRATEGIES -->
          <p class="section-divider">StrategyProfile (pesi ruolo e vincoli soft)</p>

          <div class="check-col" role="group" aria-label="Strategie da eseguire">
            @for (s of availableStrategies(); track s) {
              <label class="strategy-check" [class.active]="selectedStrategies().has(s)">
                <input type="checkbox" [checked]="selectedStrategies().has(s)"
                       (change)="toggleStrategy(s)" />
                <span>{{ meta(s).icon }}</span>
                <span>{{ meta(s).label }}</span>
              </label>
            }
          </div>

          @if (singleStrategySelected()) {
            <button class="advanced-toggle" style="margin-top:4px"
                    (click)="showCustomWeights.set(!showCustomWeights())">
              Personalizza pesi per ruolo {{ showCustomWeights() ? '▲' : '▼' }}
            </button>
            @if (showCustomWeights()) {
              <app-field-legend
                fieldId="legend-customWeights"
                [description]="OPTIMIZER_LEGENDS['customWeights'].description"
                [examples]="OPTIMIZER_LEGENDS['customWeights'].examples" />
              <div class="custom-weights-panel">
                @for (role of ['P','D','C','A']; track role) {
                  <div class="field-group">
                    <label class="field-label" style="display:flex;justify-content:space-between">
                      <span>Peso ruolo {{ roleLabel(role) }} ({{ role }})</span>
                      <span class="field-hint">{{ customWeights()[role] | number:'1.2-2' }}</span>
                    </label>
                    <input type="range" min="0.1" max="3" step="0.05"
                           [ngModel]="customWeights()[role]"
                           (ngModelChange)="setCustomWeight(role, $event)"
                           [attr.aria-label]="'Peso per il ruolo ' + roleLabel(role)" />
                  </div>
                }
              </div>
            }
          }

          <button class="run-btn" (click)="run()" [disabled]="running() || !canRun()">
            @if (running()) {
              <span class="spinner"></span>
              @if (jobStatus(); as js) {
                Job {{ js }}{{ jobId() ? ' · ' + jobId()!.slice(0, 8) : '' }}…
              } @else {
                Ottimizzazione in corso…
              }
            } @else {
              Esegui ottimizzatore
              @if (monteCarloEnabled() && nSimulations() > 25) {
                <span class="field-hint"> (async N&gt;25)</span>
              }
            }
          </button>
          @if (usedAsyncJob() && jobStatus()) {
            <p class="muted job-hint" role="status">
              Esecuzione asincrona: il server elabora SAA in background e questa UI fa polling.
            </p>
          }

          @if (error()) {
            <app-error-boundary title="Errore ottimizzatore" [message]="error()!" />
          }
        </aside>

        <!-- ── Results panel ─────────────────────────────── -->
        <section class="results-panel">
          @if (!results()) {
            @if (running()) {
              <div class="results-placeholder">
                <div style="width:100%;max-width:480px;display:flex;flex-direction:column;gap:12px">
                  @for (_ of [1,2,3,4]; track $index) {
                    <app-skeleton height="120px" />
                  }
                </div>
              </div>
            } @else {
              <div class="results-placeholder">
                <div class="placeholder-icon">🏗️</div>
                <p class="placeholder-text">Configura i parametri e avvia l'ottimizzatore per vedere la rosa consigliata</p>
              </div>
            }
          } @else {
            <div class="strategy-tabs" role="tablist" aria-label="Risultati per strategia">
              @for (name of resultKeys(); track name) {
                <button class="strategy-tab"
                        [class.active]="activeStrategy() === name"
                        (click)="activeStrategy.set(name)"
                        [attr.role]="'tab'"
                        [attr.aria-selected]="activeStrategy() === name">
                  <span>{{ meta(name).icon }}</span>
                  <span>{{ meta(name).label }}</span>
                  @if (resultFor(name); as r) {
                    <span class="tab-score" title="Punteggio proiettato totale della rosa">Score {{ r.totalProjectedScore | number:'1.1-1' }}</span>
                  }
                </button>
              }
            </div>

            
            @if (results()?.diversity; as div) {
              <div class="insight-banner" [class.insight-banner--warn]="div.lowDiversity" role="status">
                <strong>Diversità strategie</strong>
                · Jaccard medio {{ div.meanPairwiseJaccard | number:'1.2-2' }}
                @if (div.lowDiversity) {
                  <span class="badge badge--warn">bassa diversità — le rose sono quasi uguali</span>
                } @else {
                  <span class="badge badge--ok">ok</span>
                }
              </div>
            }

            @if (multiMcSummary(); as mc) {
              <div class="mc-summary card" aria-label="Monte Carlo summary">
                <div class="mc-summary__header">
                  <h3 class="mc-summary__title">Monte Carlo</h3>
                  <span class="badge">{{ mc.mode }} · N={{ mc.nSimulations }}</span>
                  @if (mc.scenariosCompleted != null) {
                    <span class="muted">scenari {{ mc.scenariosCompleted }}</span>
                  }
                  @if (mc.wallTimeSeconds != null) {
                    <span class="muted">{{ mc.wallTimeSeconds | number:'1.2-2' }}s</span>
                  }
                </div>
                <div class="mc-summary__stats">
                  <div class="stat-card">
                    <p class="stat-label">Stability index</p>
                    <p class="stat-value">{{ mc.stabilityIndex ?? 0 | number:'1.2-2' }}</p>
                  </div>
                  <div class="stat-card">
                    <p class="stat-label">Jaccard scenari</p>
                    <p class="stat-value">{{ mc.meanPairwiseJaccard ?? 0 | number:'1.2-2' }}</p>
                  </div>
                  @if (mc.yieldStability?.probAboveThreshold != null) {
                    <div class="stat-card">
                      <p class="stat-label">P(yield ≥ soglia)</p>
                      <p class="stat-value">{{ mc.yieldStability!.probAboveThreshold | percent:'1.0-0' }}</p>
                    </div>
                  }
                </div>
                @if (topSelectionFrequency(); as freqRows) {
                  <div class="freq-table-wrap">
                    <p class="stat-label">Frequenza selezione (top)</p>
                    <table class="freq-table">
                      <thead><tr><th>Player</th><th>Freq</th></tr></thead>
                      <tbody>
                        @for (row of freqRows; track row.id) {
                          <tr>
                            <td [title]="row.id">{{ row.name }}</td>
                            <td>
                              <div class="freq-bar-track" aria-hidden="true">
                                <div class="freq-bar-fill" [style.width.%]="row.freq * 100"></div>
                              </div>
                              {{ row.freq | percent:'1.0-0' }}
                            </td>
                          </tr>
                        }
                      </tbody>
                    </table>
                  </div>
                }
                @if (mc.warnings?.length) {
                  <ul class="mc-warnings">
                    @for (w of mc.warnings; track w) {
                      <li>{{ w }}</li>
                    }
                  </ul>
                }
              </div>
            }

            @if (activeResult(); as r) {
              <div class="summary-row">
                <div class="stat-card">
                  <p class="stat-label">Punteggio proiettato totale</p>
                  <p class="stat-value">{{ r.totalProjectedScore | number:'1.2-2' }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Costo nominale (somma listini)</p>
                  <p class="stat-value">{{ r.totalNominalCost }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Costo effettivo (post-inflazione)</p>
                  <p class="stat-value">{{ r.totalEffectiveCost | number:'1.1-1' }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Residuo di budget</p>
                  <p class="stat-value">{{ r.budgetResidual | number:'1.0-0' }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Squadre di Serie A in rosa</p>
                  <p class="stat-value">{{ r.distinctTeamsCount }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Stato del risolutore</p>
                  <p class="stat-value" [class.text-green]="r.status === 'Optimal'">{{ r.status === 'Optimal' ? 'Ottimale' : r.status }}</p>
                </div>
                @if (r.winProbability != null) {
                  <div class="stat-card" title="Stima Monte Carlo: probabilità che la rosa rimanga entro il budget dopo l'asta">
                    <p class="stat-label">Probabilità successo budget</p>
                    <p class="stat-value" [class.text-green]="r.winProbability >= 0.7"
                       [style.color]="r.winProbability < 0.4 ? '#EF4444' : r.winProbability < 0.7 ? '#F59E0B' : ''">
                      {{ r.winProbability | percent:'1.0-0' }}
                    </p>
                  </div>
                }
              </div>

              <div class="formation-row">
                <p class="section-label">Fattibilità moduli tattici</p>
                <div class="formation-chips">
                  @for (entry of formationEntries(r); track entry[0]) {
                    <span class="formation-chip" [class.ok]="entry[1]"
                          [title]="(entry[1] ? 'Modulo giocabile' : 'Modulo NON giocabile con questa rosa')">
                      {{ entry[0] }} {{ entry[1] ? '✓' : '✗' }}
                    </span>
                  }
                </div>
              </div>


              @if (r.nearOptimal?.length) {
                <div class="near-optimal card" aria-label="Alternative near-optimal">
                  <h3 class="mc-summary__title">Alternative near-optimal</h3>
                  <p class="muted">Rose ricalcolate escludendo i top scorer della soluzione primaria.</p>
                  <div class="near-optimal__list">
                    @for (alt of r.nearOptimal; track $index) {
                      <details class="near-optimal__item">
                        <summary>
                          Δ score {{ alt.scoreDelta | number:'1.2-2' }}
                          ({{ alt.scoreDeltaPct | percent:'1.1-1' }})
                          · esclusi: {{ alt.excludedPlayerIds.join(', ') }}
                        </summary>
                        <ul class="near-optimal__squad">
                          @for (pl of alt.squad; track pl.playerId) {
                            <li>{{ pl.name }} <span class="muted">({{ pl.role }} · {{ pl.projectedScore | number:'1.1-1' }})</span></li>
                          }
                        </ul>
                      </details>
                    }
                  </div>
                </div>
              }

              <div class="role-breakdown">
                @for (role of ['P','D','C','A']; track role) {
                  <div class="role-strip" [style.border-color]="roleColor(role)"
                       [title]="'Numero di giocatori nel ruolo ' + roleLabel(role)">
                    <span class="role-label" [style.color]="roleColor(role)">{{ roleLabel(role) }}</span>
                    <span class="role-count">{{ r.roleBreakdown[role] || 0 }}</span>
                  </div>
                }
              </div>

              <div class="squad-table-wrap">
                <table class="squad-table">
                  <thead>
                    <tr>
                      <th>Ruolo</th>
                      <th>Giocatore</th>
                      <th class="hide-sm">Squadra</th>
                      <th class="num">Costo</th>
                      <th class="num hide-sm">Eff.</th>
                      <th class="num">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    @for (p of sortedSquad(r); track p.playerId) {
                      <tr class="clickable-row"
                          (click)="selectedPlayer.set(p)"
                          (keydown.enter)="selectedPlayer.set(p)"
                          (keydown.space)="$event.preventDefault(); selectedPlayer.set(p)"
                          tabindex="0"
                          role="button"
                          [attr.aria-label]="'Dettaglio ' + p.name">
                        <td>
                          <span class="role-badge" [style.color]="roleColor(p.role)"
                                [style.border-color]="roleColor(p.role)">
                            {{ roleLabel(p.role) }}
                          </span>
                        </td>
                        <td class="player-name">{{ p.name }}</td>
                        <td class="team-name hide-sm">{{ p.realTeam }}</td>
                        <td class="num" title="Costo nominale (listino base)">{{ p.cost }}</td>
                        <td class="num faded hide-sm" title="Costo effettivo post-inflazione">{{ p.effectiveCost | number:'1.1-1' }}</td>
                        <td class="num accent" title="Punteggio proiettato per il giocatore">{{ p.projectedScore | number:'1.2-2' }}</td>
                      </tr>
                    }
                  </tbody>
                </table>
              </div>
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
    .optimizer-page { display:flex; flex-direction:column; height:100%; overflow:hidden; }
    .page-header { padding:16px; border-bottom:1px solid var(--color-border); flex-shrink:0; }
    @media (min-width: 640px) { .page-header { padding:20px 24px 16px; } }
    .page-title { font-size:16px; font-weight:700; color:var(--color-text-primary); margin:0; }
    @media (min-width: 640px) { .page-title { font-size:18px; } }
    .page-subtitle { font-size:11px; color:var(--color-text-secondary); margin:2px 0 0; }
    @media (min-width: 640px) { .page-subtitle { font-size:12px; } }

    .optimizer-body {
      display:flex; flex-direction:column;
      flex:1; overflow:hidden; min-height:0;
    }
    @media (min-width: 768px) {
      .optimizer-body {
        display:grid; grid-template-columns:300px 1fr;
      }
    }

    /* Config panel */
    .config-panel {
      border-radius:0; border-top:none; border-bottom:1px solid var(--color-border);
      border-left:none; border-right:none;
      padding:16px; overflow-y:auto; max-height:50vh;
      display:flex; flex-direction:column; gap:10px;
    }
    @media (min-width: 768px) {
      .config-panel {
        max-height:none; border-bottom:none;
        border-right:1px solid var(--color-border);
      }
    }
    .section-divider {
      font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:0.08em; color:var(--color-text-secondary);
      margin:6px 0 0; padding-bottom:6px;
      border-bottom:1px solid var(--color-border);
    }
    .field-group { display:flex; flex-direction:column; gap:4px; min-width:0; }
    .field-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .field-label { font-size:11px; font-weight:500; color:var(--color-text-secondary); }
    .preset-description {
      margin: 6px 0 0;
      font-size: 12px;
      line-height: 1.45;
      color: var(--color-text-secondary);
    }
    .preset-description.muted { opacity: 0.85; }

    .field-hint { font-size:10px; opacity:0.6; }
    .field-input {
      background:var(--color-bg); border:1px solid var(--color-border);
      border-radius:6px; padding:8px;
      color:var(--color-text-primary); font-size:13px;
      outline:none; width:100%; min-width:0;
    }
    @media (min-width: 640px) {
      .field-input { padding:6px 8px; font-size:12px; }
    }
    .field-input:focus { border-color:var(--color-accent); }
    .field-textarea { resize:vertical; font-family:var(--font-sans); }

    /* Formations check grid */
    .check-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .check-chip {
      display:flex; align-items:center; justify-content:center;
      padding:8px; border-radius:6px;
      border:1px solid var(--color-border); background:var(--color-bg);
      cursor:pointer; font-size:12px; font-weight:500;
      color:var(--color-text-secondary);
      transition:border-color 100ms, color 100ms;
      min-height:36px;
    }
    .check-chip.active {
      border-color:var(--color-accent); color:var(--color-text-primary);
      background:color-mix(in srgb, var(--color-accent) 8%, transparent);
    }
    .check-chip input { display:none; }

    /* Strategies column */
    .check-col { display:flex; flex-direction:column; gap:5px; }
    .strategy-check {
      display:flex; align-items:center; gap:8px;
      padding:8px 10px; border-radius:6px;
      border:1px solid var(--color-border); background:var(--color-bg);
      cursor:pointer; font-size:12px; color:var(--color-text-secondary);
      transition:border-color 100ms, color 100ms;
      min-height:36px;
    }
    .strategy-check.active {
      border-color:var(--color-accent); color:var(--color-text-primary);
      background:color-mix(in srgb, var(--color-accent) 8%, transparent);
    }
    .strategy-check input { display:none; }

    .run-btn {
      margin-top:4px; width:100%; padding:10px;
      border-radius:8px; background:var(--color-accent);
      color:#fff; font-size:13px; font-weight:600;
      border:none; cursor:pointer;
      display:flex; align-items:center; justify-content:center; gap:6px;
      transition:opacity 120ms;
      min-height:40px;
    }
    .run-btn:disabled { opacity:0.5; cursor:not-allowed; }
    .run-btn:not(:disabled):hover { opacity:0.9; }
    .spinner {
      width:14px; height:14px;
      border:2px solid rgba(255,255,255,0.3);
      border-top-color:#fff; border-radius:50%;
      animation:spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform:rotate(360deg); } }

    /* Results panel */
    .results-panel {
      display:flex; flex-direction:column; overflow:hidden;
      min-height:0;
    }
    @media (min-width: 768px) {
      .results-panel { border-left:none; }
    }
    .results-placeholder {
      flex:1; display:flex; flex-direction:column;
      align-items:center; justify-content:center;
      gap:12px; padding:24px 16px; overflow-y:auto;
    }
    @media (min-width: 640px) { .results-placeholder { padding:32px; } }
    .placeholder-icon { font-size:40px; }
    .placeholder-text {
      font-size:13px; color:var(--color-text-secondary);
      text-align:center; max-width:280px;
    }

    .strategy-tabs {
      display:flex; border-bottom:1px solid var(--color-border);
      flex-shrink:0; padding:0 12px; overflow-x:auto;
      -webkit-overflow-scrolling:touch;
    }
    @media (min-width: 640px) { .strategy-tabs { padding:0 16px; } }
    .strategy-tab {
      display:flex; align-items:center; gap:6px;
      padding:12px 10px; border:none; background:none;
      color:var(--color-text-secondary); font-size:12px; font-weight:500;
      border-bottom:2px solid transparent; cursor:pointer;
      white-space:nowrap; transition:color 100ms, border-color 100ms;
      min-height:44px;
    }
    @media (min-width: 640px) { .strategy-tab { padding:12px 14px; } }
    .strategy-tab.active { color:var(--color-accent); border-bottom-color:var(--color-accent); }
    .tab-score {
      background:var(--color-surface-raised); border-radius:9999px;
      padding:1px 7px; font-size:11px; color:var(--color-text-secondary);
    }

    .summary-row {
      display:grid; grid-template-columns:repeat(2,1fr);
      border-bottom:1px solid var(--color-border); flex-shrink:0;
    }
    @media (min-width: 640px) { .summary-row { grid-template-columns:repeat(3,1fr); } }
    @media (min-width: 1024px) { .summary-row { grid-template-columns:repeat(6,1fr); } }
    .stat-card {
      padding:10px 12px;
      border-right:1px solid var(--color-border);
      border-bottom:1px solid var(--color-border);
    }
    @media (min-width: 640px) {
      .stat-card:nth-child(2n) { border-right:none; }
      .stat-card:nth-last-child(-n+2) { border-bottom:none; }
    }
    @media (min-width: 1024px) {
      .stat-card { padding:12px 14px; border-bottom:none; }
      .stat-card { border-right:1px solid var(--color-border); }
      .stat-card:last-child { border-right:none; }
    }
    .stat-label {
      font-size:10px; font-weight:500; text-transform:uppercase;
      letter-spacing:0.06em; color:var(--color-text-secondary); margin:0 0 3px;
    }
    .stat-value {
      font-size:14px; font-weight:700;
      font-variant-numeric:tabular-nums; color:var(--color-text-primary); margin:0;
    }
    @media (min-width: 1024px) { .stat-value { font-size:16px; } }
    .text-green { color:#22C55E !important; }

    .formation-row {
      display:flex; align-items:center; gap:10px; flex-wrap:wrap;
      padding:8px 12px; border-bottom:1px solid var(--color-border); flex-shrink:0;
    }
    @media (min-width: 640px) { .formation-row { padding:8px 16px; } }
    .section-label {
      font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:0.06em; color:var(--color-text-secondary); margin:0; white-space:nowrap;
    }
    .formation-chips { display:flex; flex-wrap:wrap; gap:4px; }
    .formation-chip {
      padding:2px 8px; border-radius:9999px; font-size:11px; font-weight:500;
      background:var(--color-surface-raised); color:var(--color-text-secondary);
      border:1px solid var(--color-border);
    }
    .formation-chip.ok {
      background:color-mix(in srgb,#22C55E 12%,transparent);
      color:#22C55E; border-color:#22C55E;
    }

    .role-breakdown {
      display:grid; grid-template-columns:repeat(2,1fr);
      flex-shrink:0; border-bottom:1px solid var(--color-border);
    }
    @media (min-width: 640px) { .role-breakdown { grid-template-columns:repeat(4,1fr); } }
    .role-strip {
      display:flex; align-items:center; justify-content:space-between;
      padding:6px 12px;
      border-right:1px solid var(--color-border);
      border-bottom:1px solid var(--color-border);
      border-left:3px solid transparent;
    }
    @media (min-width: 640px) {
      .role-strip { border-bottom:none; padding:6px 14px; }
    }
    .role-strip:nth-child(2n) { border-right:none; }
    @media (min-width: 640px) { .role-strip:nth-child(2n) { border-right:1px solid var(--color-border); } }
    .role-strip:last-child { border-right:none; }
    .role-label { font-size:11px; font-weight:600; }
    .role-count { font-size:14px; font-weight:700; color:var(--color-text-primary); }

    .squad-table-wrap {
      flex:1; overflow:auto; min-height:0;
      margin:0 -12px;
    }
    @media (min-width: 640px) { .squad-table-wrap { margin:0 -16px; } }
    .squad-table { width:100%; min-width:520px; border-collapse:collapse; font-size:13px; }
    .hide-sm { display:none; }
    @media (min-width: 640px) { .hide-sm { display:table-cell; } }
    .squad-table thead th {
      position:sticky; top:0; z-index:1;
      background:var(--color-surface); padding:8px 10px; text-align:left;
      font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:0.05em; color:var(--color-text-secondary);
      border-bottom:1px solid var(--color-border);
    }
    @media (min-width: 640px) { .squad-table thead th { padding:8px 14px; } }
    .squad-table tbody tr {
      border-bottom:1px solid var(--color-border); transition:background 100ms;
    }
    .squad-table tbody tr:hover { background:var(--color-surface-raised); }
    .squad-table tbody td { padding:8px 10px; color:var(--color-text-primary); }
    @media (min-width: 640px) { .squad-table tbody td { padding:8px 14px; } }
    .squad-table .num { text-align:right; font-variant-numeric:tabular-nums; }
    .role-badge {
      display:inline-flex; align-items:center; justify-content:center;
      width:36px; padding:1px 0; border-radius:4px;
      border:1px solid; font-size:10px; font-weight:700;
    }
    .player-name { font-weight:500; }
    .team-name { color:var(--color-text-secondary); font-size:12px; }
    .faded { color:var(--color-text-secondary); }
    .accent { color:var(--color-accent); font-weight:600; }

    /* Mobile (< md): collapse the split-pane app-shell layout (fixed-height
       page + independently-scrolling config/results panels) into one
       normal flowing page instead of nested scroll boxes. Placed last so
       it wins over the min-width overrides above below the md breakpoint. */
    @media (max-width: 767px) {
      .optimizer-page { height: auto; overflow: visible; }
      .optimizer-body { overflow: visible; min-height: auto; }
      .config-panel { max-height: none; overflow-y: visible; }
      .results-panel { overflow: visible; min-height: auto; }
      .results-placeholder { overflow-y: visible; }
    }

    .clickable-row { cursor: pointer; }
    .clickable-row:hover { background: var(--color-surface-raised); }
    .clickable-row:focus-visible { outline: 2px solid var(--color-brand-500, #6366f1); outline-offset: -2px; }

    /* Monte Carlo / diversity insights */
    .insight-banner {
      display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px;
      padding: 10px 14px; margin-bottom: 12px;
      border-radius: var(--radius-md, 8px);
      background: var(--color-surface-2, #1a1a22);
      border: 1px solid var(--color-border, #2a2a35);
      font-size: 0.875rem;
    }
    .insight-banner--warn {
      border-color: #F59E0B55;
      background: #F59E0B12;
    }
    .badge {
      display: inline-flex; align-items: center;
      padding: 2px 8px; border-radius: 999px;
      font-size: 0.75rem; font-weight: 600;
      background: var(--color-surface-3, #252530);
    }
    .badge--warn { background: #F59E0B33; color: #FBBF24; }
    .badge--ok { background: #10B98133; color: #34D399; }
    .field-warning {
      margin: 0 0 12px; padding: 8px 12px;
      border-radius: 6px; font-size: 0.8rem;
      background: #F59E0B18; border: 1px solid #F59E0B44; color: #FBBF24;
    }
    .field-group--toggle .field-label {
      display: flex; align-items: center; gap: 8px; cursor: pointer;
    }
    .mc-summary {
      padding: 14px 16px; margin-bottom: 14px;
    }
    .mc-summary__header {
      display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px;
      margin-bottom: 10px;
    }
    .mc-summary__title {
      margin: 0; font-size: 1rem; font-weight: 600;
    }
    .mc-summary__stats {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px; margin-bottom: 12px;
    }
    .freq-table-wrap { margin-top: 8px; }
    .freq-table {
      width: 100%; border-collapse: collapse; font-size: 0.8125rem;
    }
    .freq-table th, .freq-table td {
      padding: 6px 8px; text-align: left;
      border-bottom: 1px solid var(--color-border, #2a2a35);
    }
    .freq-bar-track {
      display: inline-block; width: 72px; height: 6px;
      margin-right: 8px; vertical-align: middle;
      background: var(--color-surface-3, #252530); border-radius: 3px;
      overflow: hidden;
    }
    .freq-bar-fill {
      height: 100%; background: var(--color-accent, #6366f1); border-radius: 3px;
    }
    .mc-warnings {
      margin: 8px 0 0; padding-left: 18px; font-size: 0.75rem;
      color: var(--color-text-secondary, #a1a1aa);
    }
    .near-optimal { padding: 14px 16px; margin: 12px 0; }
    .near-optimal__item { margin: 6px 0; }
    .near-optimal__squad {
      margin: 6px 0 0; padding-left: 18px; font-size: 0.8125rem;
    }
    .muted { color: var(--color-text-secondary, #a1a1aa); font-size: 0.8125rem; }
    .job-hint { margin: 8px 0 0; }

  `],
})
export class OptimizerComponent {
  private readonly optimizerService = inject(OptimizerService);
  private readonly quotationService = inject(QuotationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly allFormations = ALL_FORMATIONS;
  /** Legende dei campi del configuratore, esposte al template. */
  protected readonly OPTIMIZER_LEGENDS = OPTIMIZER_LEGENDS;

  /** Preset catalog (immutable). Exposed for the template select. */
  readonly presets: readonly OptimizerPreset[] = OPTIMIZER_PRESETS;
  protected readonly OPTIMIZER_PRESET_NONE = OPTIMIZER_PRESET_NONE;

  /**
   * Currently selected preset id. Empty string = operator-driven custom config.
   * Applying a preset patches form signals; it does not auto-run the solver.
   */
  readonly selectedPlayer = signal<SquadPlayer | null>(null);

  readonly selectedPresetId = signal<string>(OPTIMIZER_PRESET_NONE);
  readonly activePreset = computed(() => findOptimizerPreset(this.selectedPresetId()));

  // Strategies loaded from API; fallback to known names if unavailable
  readonly availableStrategies = signal<string[]>(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']);

  // ── Basic ─────────────────────────────────────────────
  readonly seasons = signal<number[]>([]);
  readonly seasonsLoading = signal(true);
  readonly seasonStart = signal<number>(2024);
  readonly budget = signal(500);
  readonly numParticipants = signal(8);
  readonly minQtA = signal(1);
  readonly solverTimeoutSeconds = signal(30);

  // ── Squad constraints ─────────────────────────────────
  readonly minDistinctTeams = signal(12);
  readonly maxPlayersPerTeam = signal(4);
  readonly bigTeamsCap = signal(10);
  readonly bigTeamsRaw = signal('Inter, Milan, Juventus, Napoli');
  readonly maxSinglePlayerBudgetShare = signal(0.30);

  // ── Must include / exclude ─────────────────────────────
  readonly mustIncludeRaw = signal('');
  readonly excludeRaw = signal('');

  // ── Formations ────────────────────────────────────────
  readonly selectedFormations = signal(new Set(ALL_FORMATIONS.map(f => f.label)));
  readonly preferredFormationLabel = signal<string>('');

  // ── Ruleset ───────────────────────────────────────────
  readonly ruleset = signal<'CLASSIC' | 'MANTRA'>('CLASSIC');

  // ── Inflation model ───────────────────────────────────
  readonly inflationPercentileThreshold = signal(0.7);
  readonly maxInflationMultiplier = signal(1.6);
  readonly baseInflationRate = signal(0.05);
  readonly baselineParticipants = signal(8);
  /** Club Elo team-strength adjustment on the effective cost. 0 = disabled
   *  (matches the backend Pydantic default; see `InflationConfigSchema.team_strength_multiplier`). */
  readonly teamStrengthMultiplier = signal(0.0);

  // ── Risk & VAR ────────────────────────────────────────
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

  // ── Strategies ────────────────────────────────────────
  readonly selectedStrategies = signal(new Set(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']));
  readonly strategyProfilesMap = signal<Record<string, StrategyProfile>>({});
  readonly showCustomWeights = signal(false);
  readonly customWeights = signal<Record<string, number>>({ P: 1, D: 1, C: 1, A: 1 });

  readonly singleStrategySelected = computed(() => this.selectedStrategies().size === 1);

  // ── Results ───────────────────────────────────────────
  readonly running = signal(false);
  /** Async job progress label (queued|running|…). Empty when sync. */
  readonly jobStatus = signal<string | null>(null);
  readonly jobId = signal<string | null>(null);
  readonly usedAsyncJob = signal(false);

  readonly error = signal<string | null>(null);
  readonly results = signal<MultiStrategyResult | null>(null);
  readonly activeStrategy = signal<string>('');

  readonly resultKeys = computed(() => Object.keys(this.results()?.results ?? {}));
  /** Top-level or per-strategy MC summary from last multi response. */
  readonly multiMcSummary = computed(() => {
    const res = this.results();
    if (!res) return null;
    return res.monteCarloSummary
      ?? res.results[this.activeStrategy()]?.monteCarloSummary
      ?? null;
  });
  /** playerId → display name from all squads in the last result set. */
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

  /**
   * Handles preset select changes. Applying a preset never clears
   * operator-owned inputs (season, mustInclude, exclude).
   */
  onPresetChange(presetId: string): void {
    this.selectedPresetId.set(presetId ?? OPTIMIZER_PRESET_NONE);
    const preset = findOptimizerPreset(presetId);
    if (preset) {
      this.applyPreset(preset);
    }
  }

  /**
   * Patches form signals from a preset request payload.
   * Intentionally pure w.r.t. results / running state.
   */
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
    // null = clear filter; number = set threshold
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

    // Strategies: customStrategies takes precedence over strategyNames (API contract).
    if (req.customStrategies?.length) {
      const names = req.customStrategies.map(s => s.name);
      this.selectedStrategies.set(new Set(names));
      const primary = req.customStrategies[0];
      this.customWeights.set({ P: 1, D: 1, C: 1, A: 1, ...primary.roleWeight });
      this.showCustomWeights.set(true);
      // Merge into local profile map so subsequent custom edits keep constraints.
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
    // Manual strategy edits leave the preset in "custom" mode so the select
    // reflects that the form no longer matches a canned profile.
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

  /**
   * Enqueue MC job and poll until completed/failed or max attempts.
   * Maps the single-strategy job result into MultiStrategyResult for the existing UI.
   */
  private _runAsyncJob(req: OptimizationRequest): void {
    // Prefer customStrategies (body wins on API); else first selected default name.
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
          // Exhausted polls without terminal state
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

  /** Build OptimizationRequest from current form signals (single source of truth). */
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
