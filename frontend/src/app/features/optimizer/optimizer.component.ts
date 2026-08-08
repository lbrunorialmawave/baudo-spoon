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
import { SEASON_FALLBACK_LIST } from '../../core/constants/season-fallback.constant';
import {
  FormationConfig,
  MultiStrategyResult,
  OptimizationRequest,
  OptimizationResult,
  ParetoPoint,
  ParetoResponse,
  SensitivityResponse,
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
import { MANTRA_MODULE_LABELS } from '../../core/constants/shared-presets';
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
  riskLambda: {
    description: 'Solo in modalità mean_std: l’obiettivo diventa mean − λ·std. λ più alto penalizza di più la volatilità degli scenari Monte Carlo.',
    examples: [
      { label: '0.0', value: 'solo media (nessuna penalità)' },
      { label: '0.5', value: 'default equilibrato' },
      { label: '1.0–1.5', value: 'molto prudente' },
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
  hybridBlend: {
    description: 'Peso in [0,1] del segnale MANTRA-ibrido (fpIbrido) nella funzione obiettivo CLASSIC, stessa forma di varBlend. 0 = disattivato (default). Richiede artefatto mantra_ibrido; i giocatori senza match mantengono lo score base.',
    examples: [
      { label: '0.0', value: 'off — comportamento legacy (default)' },
      { label: '0.3–0.5', value: 'blend moderato con pilastri MANTRA' },
      { label: '0.8–1.0', value: 'forte peso su fpIbrido' },
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

      <!-- Sticky top bar: identity + preset + primary CTA -->
      <header class="opt-topbar">
        <div class="opt-topbar__brand">
          <h1 class="opt-topbar__title">Ottimizzatore rosa</h1>
          <p class="opt-topbar__subtitle">Trova la rosa migliore entro budget, vincoli e rischio</p>
        </div>

        <div class="opt-topbar__actions">
          <div class="opt-preset">
            <label class="sr-only" for="opt-preset">Preset</label>
            <select
              id="opt-preset"
              class="field-input field-input--compact"
              [ngModel]="selectedPresetId()"
              (ngModelChange)="onPresetChange($event)"
              [attr.aria-describedby]="'legend-preset'"
            >
              <option [ngValue]="OPTIMIZER_PRESET_NONE">Personalizzato</option>
              @for (p of presets; track p.id) {
                <option [ngValue]="p.id">{{ p.labelIt }}</option>
              }
            </select>
          </div>
          <button
            type="button"
            class="btn btn--primary btn--run"
            (click)="run(); mobilePane.set('results')"
            [disabled]="running() || !canRun()"
          >
            @if (running()) {
              <span class="spinner" aria-hidden="true"></span>
              <span>
                @if (jobStatus(); as js) {
                  {{ js }}…
                } @else {
                  Calcolo…
                }
              </span>
            } @else {
              Ottimizza rosa
            }
          </button>
        </div>
      </header>

      @if (activePreset(); as preset) {
        <p class="opt-preset-banner" id="legend-preset">{{ preset.description }}</p>
      } @else {
        <p class="opt-preset-banner muted" id="legend-preset">
          Scegli un preset per precompilare vincoli e leve, oppure configura a mano.
        </p>
      }

      <!-- Mobile pane switcher -->
      <nav class="opt-pane-nav" aria-label="Sezioni">
        <button type="button" class="opt-pane-nav__btn"
                [class.active]="mobilePane() === 'config'"
                (click)="mobilePane.set('config')">1 · Configura</button>
        <button type="button" class="opt-pane-nav__btn"
                [class.active]="mobilePane() === 'results'"
                (click)="mobilePane.set('results')">
          2 · Risultati
          @if (results()) {
            <span class="opt-pane-nav__badge">{{ resultKeys().length }}</span>
          }
        </button>
        <button type="button" class="opt-pane-nav__btn"
                [class.active]="mobilePane() === 'analysis'"
                (click)="mobilePane.set('analysis')">3 · Analisi</button>
      </nav>

      <div class="opt-shell">

        <!-- ════════ CONFIG ════════ -->
        <aside class="opt-pane opt-config" [class.opt-pane--active]="mobilePane() === 'config'">

          <!-- Essentials -->
          <section class="opt-section" [class.open]="isSectionOpen('essentials')">
            <button type="button" class="opt-section__head" (click)="toggleSection('essentials')"
                    [attr.aria-expanded]="isSectionOpen('essentials')">
              <span class="opt-section__title">Essenziali</span>
              <span class="opt-section__hint">stagione, budget, regole</span>
              <span class="opt-section__chev" aria-hidden="true"></span>
            </button>
            @if (isSectionOpen('essentials')) {
              <div class="opt-section__body">
                <div class="field-group">
                  <label class="field-label" for="opt-seasonStart">Stagione</label>
                  @if (seasonsLoading()) {
                    <app-skeleton height="40px" />
                  } @else {
                    <select id="opt-seasonStart" class="field-input" [(ngModel)]="seasonStart"
                            [attr.aria-describedby]="'legend-seasonStart'">
                      @for (s of seasons(); track s) {
                        <option [value]="s">{{ s }}/{{ s + 1 }}</option>
                      }
                    </select>
                  }
                  <app-field-legend fieldId="legend-seasonStart"
                    [description]="OPTIMIZER_LEGENDS['seasonStart'].description"
                    [examples]="OPTIMIZER_LEGENDS['seasonStart'].examples" />
                </div>

                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-budget">Budget <span class="field-hint">crediti</span></label>
                    <input id="opt-budget" class="field-input" type="number" min="200" max="1000" step="25"
                           [(ngModel)]="budget" [attr.aria-describedby]="'legend-budget'" />
                    <app-field-legend fieldId="legend-budget"
                      [description]="OPTIMIZER_LEGENDS['budget'].description"
                      [examples]="OPTIMIZER_LEGENDS['budget'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-numParticipants">Partecipanti lega</label>
                    <input id="opt-numParticipants" class="field-input" type="number" min="4" max="16" step="1"
                           [(ngModel)]="numParticipants" [attr.aria-describedby]="'legend-numParticipants'" />
                    <app-field-legend fieldId="legend-numParticipants"
                      [description]="OPTIMIZER_LEGENDS['numParticipants'].description"
                      [examples]="OPTIMIZER_LEGENDS['numParticipants'].examples" />
                  </div>
                </div>

                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-ruleset">Regolamento</label>
                    <select id="opt-ruleset" class="field-input" [(ngModel)]="ruleset"
                            [attr.aria-describedby]="'legend-ruleset'">
                      <option value="CLASSIC">CLASSIC — P / D / C / A</option>
                      <option value="MANTRA">MANTRA — 12 ruoli</option>
                    </select>
                    <app-field-legend fieldId="legend-ruleset"
                      [description]="OPTIMIZER_LEGENDS['ruleset'].description"
                      [examples]="OPTIMIZER_LEGENDS['ruleset'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-minQtA">Listino minimo <span class="field-hint">qt_a</span></label>
                    <input id="opt-minQtA" class="field-input" type="number" min="0" max="10" step="1"
                           [(ngModel)]="minQtA" [attr.aria-describedby]="'legend-minQtA'" />
                    <app-field-legend fieldId="legend-minQtA"
                      [description]="OPTIMIZER_LEGENDS['minQtA'].description"
                      [examples]="OPTIMIZER_LEGENDS['minQtA'].examples" />
                  </div>
                </div>

                <div class="field-group">
                  <span class="field-label">Strategie da confrontare</span>
                  <div class="chip-grid" role="group" aria-label="Strategie">
                    @for (name of availableStrategies(); track name) {
                      <label class="chip" [class.active]="selectedStrategies().has(name)">
                        <input type="checkbox" [checked]="selectedStrategies().has(name)"
                               (change)="toggleStrategy(name)" />
                        <span class="chip__icon" aria-hidden="true">{{ meta(name).icon }}</span>
                        <span>{{ meta(name).label }}</span>
                      </label>
                    }
                  </div>
                  @if (singleStrategySelected()) {
                    <button type="button" class="link-btn" (click)="showCustomWeights.set(!showCustomWeights())">
                      {{ showCustomWeights() ? 'Nascondi pesi ruolo' : 'Personalizza pesi ruolo' }}
                    </button>
                    @if (showCustomWeights()) {
                      <div class="weight-grid">
                        @for (role of ['P','D','C','A']; track role) {
                          <div class="field-group">
                            <label class="field-label" style="display:flex;justify-content:space-between">
                              <span>{{ roleLabel(role) }}</span>
                              <span class="field-hint">{{ customWeights()[role] | number:'1.2-2' }}</span>
                            </label>
                            <input type="range" min="0.1" max="3" step="0.05"
                                   [ngModel]="customWeights()[role]"
                                   (ngModelChange)="setCustomWeight(role, $event)"
                                   [attr.aria-label]="'Peso ' + roleLabel(role)" />
                          </div>
                        }
                      </div>
                    }
                  }
                </div>
              </div>
            }
          </section>

          <!-- Advanced config group: everything past "Essenziali" is optional
               tuning (VAR blend, ESV weight, Monte Carlo, inflation curve, ...).
               Collapsed by default so the page opens on just the essentials. -->
          <div class="opt-advanced">
            <button type="button" class="opt-advanced__toggle"
                    (click)="showAdvancedConfig.set(!showAdvancedConfig())"
                    [attr.aria-expanded]="showAdvancedConfig()">
              <span class="opt-advanced__title">Impostazioni avanzate</span>
              <span class="opt-advanced__badge">5</span>
              <span class="opt-section__chev" [class.opt-section__chev--open]="showAdvancedConfig()" aria-hidden="true"></span>
            </button>
            <p class="opt-advanced__hint">
              Vincoli extra, funzione obiettivo, robustezza, moduli/costi e filtri manuali.
              Lasciale ai valori di default se non hai esigenze specifiche.
            </p>
          </div>

          @if (showAdvancedConfig()) {

          <!-- Constraints -->
          <section class="opt-section" [class.open]="isSectionOpen('constraints')">
            <button type="button" class="opt-section__head" (click)="toggleSection('constraints')"
                    [attr.aria-expanded]="isSectionOpen('constraints')">
              <span class="opt-section__title">Vincoli rosa</span>
              <span class="opt-section__hint">club, big, share</span>
              <span class="opt-section__chev" aria-hidden="true"></span>
            </button>
            @if (isSectionOpen('constraints')) {
              <div class="opt-section__body">
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-minDistinctTeams">Min. club distinti</label>
                    <input id="opt-minDistinctTeams" class="field-input" type="number" min="1" max="25" step="1"
                           [(ngModel)]="minDistinctTeams" [attr.aria-describedby]="'legend-minDistinctTeams'" />
                    <app-field-legend fieldId="legend-minDistinctTeams"
                      [description]="OPTIMIZER_LEGENDS['minDistinctTeams'].description"
                      [examples]="OPTIMIZER_LEGENDS['minDistinctTeams'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-maxPlayersPerTeam">Max per club</label>
                    <input id="opt-maxPlayersPerTeam" class="field-input" type="number" min="1" max="10" step="1"
                           [(ngModel)]="maxPlayersPerTeam" [attr.aria-describedby]="'legend-maxPlayersPerTeam'" />
                    <app-field-legend fieldId="legend-maxPlayersPerTeam"
                      [description]="OPTIMIZER_LEGENDS['maxPlayersPerTeam'].description"
                      [examples]="OPTIMIZER_LEGENDS['maxPlayersPerTeam'].examples" />
                  </div>
                </div>
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-bigTeamsCap">Tetto big team</label>
                    <input id="opt-bigTeamsCap" class="field-input" type="number" min="0" max="25" step="1"
                           [(ngModel)]="bigTeamsCap" [attr.aria-describedby]="'legend-bigTeamsCap'" />
                    <app-field-legend fieldId="legend-bigTeamsCap"
                      [description]="OPTIMIZER_LEGENDS['bigTeamsCap'].description"
                      [examples]="OPTIMIZER_LEGENDS['bigTeamsCap'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-maxShare">Max quota su 1 giocatore</label>
                    <input id="opt-maxShare" class="field-input" type="number" min="0.05" max="1" step="0.05"
                           [(ngModel)]="maxSinglePlayerBudgetShare"
                           [attr.aria-describedby]="'legend-maxSinglePlayerBudgetShare'" />
                    <app-field-legend fieldId="legend-maxSinglePlayerBudgetShare"
                      [description]="OPTIMIZER_LEGENDS['maxSinglePlayerBudgetShare'].description"
                      [examples]="OPTIMIZER_LEGENDS['maxSinglePlayerBudgetShare'].examples" />
                  </div>
                </div>
                <div class="field-group">
                  <label class="field-label" for="opt-bigTeamsRaw">Club “big”</label>
                  <textarea id="opt-bigTeamsRaw" class="field-input field-textarea" rows="2"
                            [(ngModel)]="bigTeamsRaw" placeholder="Inter, Milan, Juventus, Napoli"
                            [attr.aria-describedby]="'legend-bigTeams'"></textarea>
                  <app-field-legend fieldId="legend-bigTeams"
                    [description]="OPTIMIZER_LEGENDS['bigTeams'].description"
                    [examples]="OPTIMIZER_LEGENDS['bigTeams'].examples" />
                </div>
              </div>
            }
          </section>

          <!-- Objective -->
          <section class="opt-section" [class.open]="isSectionOpen('objective')">
            <button type="button" class="opt-section__head" (click)="toggleSection('objective')"
                    [attr.aria-expanded]="isSectionOpen('objective')">
              <span class="opt-section__title">Funzione obiettivo</span>
              <span class="opt-section__hint">rischio, VAR, hybrid, ESV</span>
              <span class="opt-section__chev" aria-hidden="true"></span>
            </button>
            @if (isSectionOpen('objective')) {
              <div class="opt-section__body">
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-riskAversion">Risk aversion</label>
                    <input id="opt-riskAversion" class="field-input" type="number" min="0" max="5" step="0.1"
                           [(ngModel)]="riskAversion" [attr.aria-describedby]="'legend-riskAversion'" />
                    <app-field-legend fieldId="legend-riskAversion"
                      [description]="OPTIMIZER_LEGENDS['riskAversion'].description"
                      [examples]="OPTIMIZER_LEGENDS['riskAversion'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-valuationMode">Metrica base</label>
                    <select id="opt-valuationMode" class="field-input" [(ngModel)]="valuationMode"
                            [attr.aria-describedby]="'legend-valuationMode'">
                      <option value="PER_MATCH_RATING">Per partita</option>
                      <option value="SEASON_VALUE">Valore stagione</option>
                    </select>
                    <app-field-legend fieldId="legend-valuationMode"
                      [description]="OPTIMIZER_LEGENDS['valuationMode'].description"
                      [examples]="OPTIMIZER_LEGENDS['valuationMode'].examples" />
                  </div>
                </div>
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-varBlend">VAR blend</label>
                    <input id="opt-varBlend" class="field-input" type="number" min="0" max="1" step="0.1"
                           [(ngModel)]="varBlend" [attr.aria-describedby]="'legend-varBlend'" />
                    <app-field-legend fieldId="legend-varBlend"
                      [description]="OPTIMIZER_LEGENDS['varBlend'].description"
                      [examples]="OPTIMIZER_LEGENDS['varBlend'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-hybridBlend">Hybrid blend</label>
                    <input id="opt-hybridBlend" class="field-input" type="number" min="0" max="1" step="0.1"
                           [(ngModel)]="hybridBlend" [attr.aria-describedby]="'legend-hybridBlend'" />
                    <app-field-legend fieldId="legend-hybridBlend"
                      [description]="OPTIMIZER_LEGENDS['hybridBlend'].description"
                      [examples]="OPTIMIZER_LEGENDS['hybridBlend'].examples" />
                  </div>
                </div>
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-esvWeight">ESV weight</label>
                    <input id="opt-esvWeight" class="field-input" type="number" min="0" max="5" step="0.1"
                           [(ngModel)]="esvWeight" [attr.aria-describedby]="'legend-esvWeight'" />
                    <app-field-legend fieldId="legend-esvWeight"
                      [description]="OPTIMIZER_LEGENDS['esvWeight'].description"
                      [examples]="OPTIMIZER_LEGENDS['esvWeight'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-replacementMethod">Replacement level</label>
                    <select id="opt-replacementMethod" class="field-input" [(ngModel)]="replacementMethod"
                            [attr.aria-describedby]="'legend-replacementMethod'">
                      <option value="percentile">Percentile per ruolo</option>
                      <option value="roster_depth">Profondità roster</option>
                    </select>
                    <app-field-legend fieldId="legend-replacementMethod"
                      [description]="OPTIMIZER_LEGENDS['replacementMethod'].description"
                      [examples]="OPTIMIZER_LEGENDS['replacementMethod'].examples" />
                  </div>
                </div>
                <div class="field-group">
                  <label class="field-label" for="opt-minStartProb">Start probability minima <span class="field-hint">vuoto = off</span></label>
                  <input id="opt-minStartProb" class="field-input" type="number" min="0" max="1" step="0.05"
                         [ngModel]="minStartProbability() ?? ''"
                         (ngModelChange)="minStartProbability.set($event === '' || $event === null ? null : +$event)"
                         [attr.aria-describedby]="'legend-minStartProbability'" />
                  <app-field-legend fieldId="legend-minStartProbability"
                    [description]="OPTIMIZER_LEGENDS['minStartProbability'].description"
                    [examples]="OPTIMIZER_LEGENDS['minStartProbability'].examples" />
                </div>
              </div>
            }
          </section>

          <!-- Robustness -->
          <section class="opt-section" [class.open]="isSectionOpen('robustness')">
            <button type="button" class="opt-section__head" (click)="toggleSection('robustness')"
                    [attr.aria-expanded]="isSectionOpen('robustness')">
              <span class="opt-section__title">Robustezza</span>
              <span class="opt-section__hint">Monte Carlo, near-optimal</span>
              <span class="opt-section__chev" aria-hidden="true"></span>
            </button>
            @if (isSectionOpen('robustness')) {
              <div class="opt-section__body">
                <label class="toggle-row">
                  <input type="checkbox"
                         [ngModel]="monteCarloEnabled()"
                         (ngModelChange)="monteCarloEnabled.set($event)"
                         [attr.aria-describedby]="'legend-mc-enabled'" />
                  <span>
                    <strong>Monte Carlo</strong>
                    <span class="field-hint">simula scenari per stabilizzare la rosa</span>
                  </span>
                </label>
                <app-field-legend fieldId="legend-mc-enabled"
                  [description]="OPTIMIZER_LEGENDS['monteCarloEnabled'].description"
                  [examples]="OPTIMIZER_LEGENDS['monteCarloEnabled'].examples" />

                @if (monteCarloEnabled()) {
                  <div class="field-row">
                    <div class="field-group">
                      <label class="field-label" for="opt-mc-mode">Modalità</label>
                      <select id="opt-mc-mode" class="field-input" [(ngModel)]="monteCarloMode"
                              [attr.aria-describedby]="'legend-mc-mode'">
                        <option value="saa_frequency">Frequenza scenari (SAA)</option>
                        <option value="mean_std">Media − λ·deviazione</option>
                      </select>
                      <app-field-legend fieldId="legend-mc-mode"
                        [description]="OPTIMIZER_LEGENDS['monteCarloMode'].description"
                        [examples]="OPTIMIZER_LEGENDS['monteCarloMode'].examples" />
                    </div>
                    <div class="field-group">
                      <label class="field-label" for="opt-mc-n">N simulazioni</label>
                      <input id="opt-mc-n" class="field-input" type="number" min="5" max="200" step="5"
                             [(ngModel)]="nSimulations" [attr.aria-describedby]="'legend-nSimulations'" />
                      <app-field-legend fieldId="legend-nSimulations"
                        [description]="OPTIMIZER_LEGENDS['nSimulations'].description"
                        [examples]="OPTIMIZER_LEGENDS['nSimulations'].examples" />
                    </div>
                  </div>
                  @if (monteCarloMode() === 'mean_std') {
                    <div class="field-group">
                      <label class="field-label" for="opt-mc-lambda">Risk λ (mean_std)</label>
                      <input id="opt-mc-lambda" class="field-input" type="number" min="0" max="3" step="0.1"
                             [(ngModel)]="riskLambda" [attr.aria-describedby]="'legend-riskLambda'" />
                      <app-field-legend fieldId="legend-riskLambda"
                        [description]="OPTIMIZER_LEGENDS['riskLambda'].description"
                        [examples]="OPTIMIZER_LEGENDS['riskLambda'].examples" />
                    </div>
                  }
                  @if (nSimulations() > 25) {
                    <p class="inline-note">N &gt; 25: esecuzione asincrona con polling del job.</p>
                  }
                }

                <label class="toggle-row">
                  <input type="checkbox"
                         [ngModel]="nearOptimalEnabled()"
                         (ngModelChange)="nearOptimalEnabled.set($event)"
                         [attr.aria-describedby]="'legend-near-opt'" />
                  <span>
                    <strong>Alternative near-optimal</strong>
                    <span class="field-hint">esclude top scorer e ri-ottimizza</span>
                  </span>
                </label>
                <app-field-legend fieldId="legend-near-opt"
                  [description]="OPTIMIZER_LEGENDS['nearOptimal'].description"
                  [examples]="OPTIMIZER_LEGENDS['nearOptimal'].examples" />
              </div>
            }
          </section>

          <!-- Formations & inflation -->
          <section class="opt-section" [class.open]="isSectionOpen('formations')">
            <button type="button" class="opt-section__head" (click)="toggleSection('formations')"
                    [attr.aria-expanded]="isSectionOpen('formations')">
              <span class="opt-section__title">Moduli e costi</span>
              <span class="opt-section__hint">formazioni, inflazione</span>
              <span class="opt-section__chev" aria-hidden="true"></span>
            </button>
            @if (isSectionOpen('formations')) {
              <div class="opt-section__body">
                <div class="field-group">
                  <span class="field-label">Moduli ammessi</span>
                  <div class="chip-grid" role="group" aria-label="Moduli">
                    @for (f of allFormations; track f.label) {
                      <label class="chip" [class.active]="selectedFormations().has(f.label)">
                        <input type="checkbox" [checked]="selectedFormations().has(f.label)"
                               (change)="toggleFormation(f.label)" />
                        {{ f.label }}
                      </label>
                    }
                  </div>
                  <app-field-legend fieldId="legend-formations"
                    [description]="OPTIMIZER_LEGENDS['formations'].description"
                    [examples]="OPTIMIZER_LEGENDS['formations'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="opt-preferredFormation">Modulo classico imposto al solver</label>
                  <select id="opt-preferredFormation" class="field-input" [(ngModel)]="preferredFormationLabel"
                          [attr.aria-describedby]="'legend-preferredFormation'">
                    <option value="">Nessuno (solo check)</option>
                    @for (f of allFormations; track f.label) {
                      <option [value]="f.label">{{ f.label }}</option>
                    }
                  </select>
                  <app-field-legend fieldId="legend-preferredFormation"
                    [description]="OPTIMIZER_LEGENDS['preferredFormation'].description"
                    [examples]="OPTIMIZER_LEGENDS['preferredFormation'].examples" />
                </div>

                @if (ruleset() === 'MANTRA') {
                  <div class="field-group">
                    <label class="field-label" for="opt-preferredMantra">Modulo Mantra preferito</label>
                    <select id="opt-preferredMantra" class="field-input"
                            [ngModel]="preferredMantraFormation()"
                            (ngModelChange)="preferredMantraFormation.set($event)">
                      <option value="">Nessuno (solo coverage post-hoc)</option>
                      @for (lab of mantraModuleLabels; track lab) {
                        <option [value]="lab">{{ lab }}</option>
                      }
                    </select>
                    <label class="field-label field-label--inline" for="opt-enforceMantra">
                      <input id="opt-enforceMantra" type="checkbox"
                             [ngModel]="enforcePreferredMantra()"
                             (ngModelChange)="enforcePreferredMantra.set($event)" />
                      Vincolo hard ILP sul modulo Mantra
                    </label>
                  </div>
                }

                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-inflationPercentile">Soglia percentile</label>
                    <input id="opt-inflationPercentile" class="field-input" type="number" min="0" max="1" step="0.05"
                           [(ngModel)]="inflationPercentileThreshold"
                           [attr.aria-describedby]="'legend-inflationPercentileThreshold'" />
                    <app-field-legend fieldId="legend-inflationPercentileThreshold"
                      [description]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].description"
                      [examples]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-maxInflation">Max moltiplicatore</label>
                    <input id="opt-maxInflation" class="field-input" type="number" min="1" max="3" step="0.1"
                           [(ngModel)]="maxInflationMultiplier"
                           [attr.aria-describedby]="'legend-maxInflationMultiplier'" />
                    <app-field-legend fieldId="legend-maxInflationMultiplier"
                      [description]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].description"
                      [examples]="OPTIMIZER_LEGENDS['maxInflationMultiplier'].examples" />
                  </div>
                </div>
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-baseInflation">Tasso base inflazione</label>
                    <input id="opt-baseInflation" class="field-input" type="number" min="0" max="0.2" step="0.01"
                           [(ngModel)]="baseInflationRate"
                           [attr.aria-describedby]="'legend-baseInflationRate'" />
                    <app-field-legend fieldId="legend-baseInflationRate"
                      [description]="OPTIMIZER_LEGENDS['baseInflationRate'].description"
                      [examples]="OPTIMIZER_LEGENDS['baseInflationRate'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-baselineParticipants">Baseline partecipanti</label>
                    <input id="opt-baselineParticipants" class="field-input" type="number" min="2" max="20" step="1"
                           [(ngModel)]="baselineParticipants"
                           [attr.aria-describedby]="'legend-baselineParticipants'" />
                    <app-field-legend fieldId="legend-baselineParticipants"
                      [description]="OPTIMIZER_LEGENDS['baselineParticipants'].description"
                      [examples]="OPTIMIZER_LEGENDS['baselineParticipants'].examples" />
                  </div>
                </div>
                <div class="field-row">
                  <div class="field-group">
                    <label class="field-label" for="opt-teamStrength">Peso Elo club</label>
                    <input id="opt-teamStrength" class="field-input" type="number" min="0" max="1.5" step="0.05"
                           [(ngModel)]="teamStrengthMultiplier"
                           [attr.aria-describedby]="'legend-teamStrengthMultiplier'" />
                    <app-field-legend fieldId="legend-teamStrengthMultiplier"
                      [description]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].description"
                      [examples]="OPTIMIZER_LEGENDS['teamStrengthMultiplier'].examples" />
                  </div>
                  <div class="field-group">
                    <label class="field-label" for="opt-solverTimeout">Timeout solver (s)</label>
                    <input id="opt-solverTimeout" class="field-input" type="number" min="5" max="300" step="5"
                           [(ngModel)]="solverTimeoutSeconds"
                           [attr.aria-describedby]="'legend-solverTimeoutSeconds'" />
                    <app-field-legend fieldId="legend-solverTimeoutSeconds"
                      [description]="OPTIMIZER_LEGENDS['solverTimeoutSeconds'].description"
                      [examples]="OPTIMIZER_LEGENDS['solverTimeoutSeconds'].examples" />
                  </div>
                </div>
              </div>
            }
          </section>

          <!-- Filters -->
          <section class="opt-section" [class.open]="isSectionOpen('filters')">
            <button type="button" class="opt-section__head" (click)="toggleSection('filters')"
                    [attr.aria-expanded]="isSectionOpen('filters')">
              <span class="opt-section__title">Include / exclude</span>
              <span class="opt-section__hint">player id forzati</span>
              <span class="opt-section__chev" aria-hidden="true"></span>
            </button>
            @if (isSectionOpen('filters')) {
              <div class="opt-section__body">
                <div class="field-group">
                  <label class="field-label" for="opt-mustInclude">Must-include</label>
                  <textarea id="opt-mustInclude" class="field-input field-textarea" rows="2"
                            [(ngModel)]="mustIncludeRaw" placeholder="fm-12345, fm-67890"
                            [attr.aria-describedby]="'legend-mustInclude'"></textarea>
                  <app-field-legend fieldId="legend-mustInclude"
                    [description]="OPTIMIZER_LEGENDS['mustInclude'].description"
                    [examples]="OPTIMIZER_LEGENDS['mustInclude'].examples" />
                </div>
                <div class="field-group">
                  <label class="field-label" for="opt-exclude">Exclude</label>
                  <textarea id="opt-exclude" class="field-input field-textarea" rows="2"
                            [(ngModel)]="excludeRaw" placeholder="fm-12345, fm-67890"
                            [attr.aria-describedby]="'legend-exclude'"></textarea>
                  <app-field-legend fieldId="legend-exclude"
                    [description]="OPTIMIZER_LEGENDS['exclude'].description"
                    [examples]="OPTIMIZER_LEGENDS['exclude'].examples" />
                </div>
              </div>
            }
          </section>

          } <!-- /showAdvancedConfig -->

          @if (error()) {
            <app-error-boundary title="Errore ottimizzatore" [message]="error()!" />
          }
          @if (usedAsyncJob() && jobStatus()) {
            <p class="inline-note" role="status">Job asincrono: {{ jobStatus() }}{{ jobId() ? ' · ' + jobId()!.slice(0, 8) : '' }}</p>
          }
        </aside>

        <!-- ════════ RESULTS ════════ -->
        <section class="opt-pane opt-results" [class.opt-pane--active]="mobilePane() === 'results'" aria-label="Risultati">
          @if (!results()) {
            @if (running()) {
              <div class="empty-state">
                <div class="empty-state__stack">
                  @for (_ of [1,2,3]; track $index) {
                    <app-skeleton height="96px" />
                  }
                </div>
                <p class="empty-state__text">Stiamo costruendo le rose…</p>
              </div>
            } @else {
              <div class="empty-state">
                <div class="empty-state__icon" aria-hidden="true">🏗️</div>
                <h2 class="empty-state__title">Nessuna rosa ancora</h2>
                <p class="empty-state__text">
                  Imposta budget e strategie, poi tocca <strong>Ottimizza rosa</strong>.
                  Confrontiamo più profili e ti mostriamo score, costi e composizione.
                </p>
                <button type="button" class="btn btn--primary" (click)="mobilePane.set('config')">
                  Vai alla configurazione
                </button>
              </div>
            }
          } @else {
            <div class="strategy-tabs" role="tablist" aria-label="Strategia attiva">
              @for (name of resultKeys(); track name) {
                <button type="button" role="tab" class="strategy-tab"
                        [class.active]="activeStrategy() === name"
                        [attr.aria-selected]="activeStrategy() === name"
                        (click)="activeStrategy.set(name)">
                  <span aria-hidden="true">{{ meta(name).icon }}</span>
                  {{ meta(name).label }}
                </button>
              }
            </div>

            @if (resultFor(activeStrategy()); as r) {
              <div class="kpi-grid" aria-label="Indicatori chiave">
                <div class="kpi">
                  <span class="kpi__label">Score totale</span>
                  <span class="kpi__value">{{ r.totalProjectedScore | number:'1.1-1' }}</span>
                </div>
                <div class="kpi">
                  <span class="kpi__label">Costo effettivo</span>
                  <span class="kpi__value">{{ r.totalEffectiveCost | number:'1.0-0' }}</span>
                  <span class="kpi__sub">/ {{ budget() }} cr</span>
                </div>
                <div class="kpi">
                  <span class="kpi__label">Status</span>
                  <span class="kpi__value kpi__value--sm">{{ r.status }}</span>
                </div>
                @if (r.winProbability != null) {
                  <div class="kpi" title="Probabilità che la rosa resti entro budget in asta">
                    <span class="kpi__label">P(asta ok)</span>
                    <span class="kpi__value">{{ r.winProbability | percent:'1.0-0' }}</span>
                  </div>
                }
              </div>

              <div class="role-strip-row" aria-label="Composizione per ruolo">
                @for (role of ['P','D','C','A']; track role) {
                  <div class="role-strip" [style.border-color]="roleColor(role)">
                    <span class="role-strip__label" [style.color]="roleColor(role)">{{ roleLabel(role) }}</span>
                    <span class="role-strip__count">{{ r.roleBreakdown[role] || 0 }}</span>
                  </div>
                }
              </div>

              @if (formationEntries(r).length) {
                <div class="formation-row" aria-label="Schierabilità moduli classici">
                  @for (entry of formationEntries(r); track entry[0]) {
                    <span class="formation-chip" [class.ok]="entry[1]" [class.ko]="!entry[1]">
                      {{ entry[0] }} {{ entry[1] ? '✓' : '✗' }}
                    </span>
                  }
                </div>
              }
              @if (mantraFormationEntries(r).length) {
                <div class="formation-row formation-row--mantra" aria-label="Schierabilità moduli Mantra">
                  <span class="formation-row__label">Mantra</span>
                  @for (entry of mantraFormationEntries(r); track entry[0]) {
                    <span class="formation-chip" [class.ok]="entry[1]" [class.ko]="!entry[1]"
                          [title]="mantraDeficitTitle(r, entry[0])">
                      {{ entry[0] }} {{ entry[1] ? '✓' : '✗' }}
                    </span>
                  }
                </div>
              }

              @if (multiMcSummary(); as mc) {
                <div class="insight-card" aria-label="Monte Carlo">
                  <div class="insight-card__head">
                    <h3>Monte Carlo</h3>
                    <span class="muted">{{ mc.nSimulations }} scenari · {{ mc.mode }}</span>
                  </div>
                  <div class="kpi-grid kpi-grid--compact">
                    <div class="kpi">
                      <span class="kpi__label">Stability</span>
                      <span class="kpi__value">{{ (mc.stabilityIndex ?? 0) | number:'1.2-2' }}</span>
                    </div>
                    <div class="kpi">
                      <span class="kpi__label">Jaccard medio</span>
                      <span class="kpi__value">{{ (mc.meanPairwiseJaccard ?? 0) | number:'1.2-2' }}</span>
                    </div>
                  </div>
                  @if (mc.warnings?.length) {
                    <ul class="warnings-list">
                      @for (w of mc.warnings; track w) { <li>{{ w }}</li> }
                    </ul>
                  }
                </div>
              }

              @if (r.nearOptimal?.length) {
                <div class="insight-card">
                  <h3>Alternative near-optimal</h3>
                  @for (alt of r.nearOptimal; track $index) {
                    <div class="near-item">
                      <strong>Alt {{ $index + 1 }}</strong>
                      <span class="muted">score {{ alt.totalProjectedScore | number:'1.1-1' }}</span>
                    </div>
                  }
                </div>
              }

              <div class="table-card">
                <div class="table-card__head">
                  <h3>Rosa</h3>
                  <span class="muted">Tocca un giocatore per il dettaglio</span>
                </div>
                <div class="table-scroll">
                  <table class="data-table">
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
                        <tr class="row-click"
                            (click)="selectedPlayer.set(p)"
                            (keydown.enter)="selectedPlayer.set(p)"
                            (keydown.space)="$event.preventDefault(); selectedPlayer.set(p)"
                            tabindex="0"
                            role="button"
                            [attr.aria-label]="'Dettaglio ' + p.name">
                          <td>
                            <span class="role-badge" [style.color]="roleColor(p.role)"
                                  [style.border-color]="roleColor(p.role)">{{ roleLabel(p.role) }}</span>
                          </td>
                          <td class="name-cell">{{ p.name }}</td>
                          <td class="hide-sm muted">{{ p.realTeam }}</td>
                          <td class="num">{{ p.cost }}</td>
                          <td class="num hide-sm muted">{{ p.effectiveCost | number:'1.1-1' }}</td>
                          <td class="num accent">{{ p.projectedScore | number:'1.2-2' }}</td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            }
          }
        </section>

        <!-- ════════ ANALYSIS ════════ -->
        <section class="opt-pane opt-analysis" [class.opt-pane--active]="mobilePane() === 'analysis'" aria-label="Analisi">
          <div class="analysis-intro">
            <h2>Analisi avanzata</h2>
            <p class="muted">
              Usa la config attuale e <strong>una sola strategia</strong> (la prima selezionata).
              Non sostituisce il confronto multi-strategia.
            </p>
          </div>

          <div class="analysis-tabs" role="tablist">
            <button type="button" role="tab" class="analysis-tab"
                    [class.active]="analysisTab() === 'sensitivity'"
                    [attr.aria-selected]="analysisTab() === 'sensitivity'"
                    (click)="analysisTab.set('sensitivity')">Sensitivity</button>
            <button type="button" role="tab" class="analysis-tab"
                    [class.active]="analysisTab() === 'pareto'"
                    [attr.aria-selected]="analysisTab() === 'pareto'"
                    (click)="analysisTab.set('pareto')">Pareto</button>
          </div>

          @if (analysisTab() === 'sensitivity') {
            <div class="analysis-body" role="tabpanel">
              <div class="analysis-actions">
                <button type="button" class="btn btn--secondary"
                        (click)="runSensitivity()" [disabled]="analysisRunning() || !canRun()">
                  @if (analysisRunning() && analysisTab() === 'sensitivity') {
                    <span class="spinner spinner--dark"></span> Calcolo…
                  } @else {
                    Esegui sensitivity
                  }
                </button>
                <p class="muted">Sweep su risk, VAR, hybrid e budget rispetto al baseline.</p>
              </div>
              @if (analysisError() && analysisTab() === 'sensitivity') {
                <app-error-boundary title="Errore sensitivity" [message]="analysisError()!" />
              }
              @if (sensitivityResult(); as sens) {
                <div class="kpi-grid kpi-grid--compact">
                  <div class="kpi"><span class="kpi__label">Status</span><span class="kpi__value kpi__value--sm">{{ sens.baselineStatus }}</span></div>
                  <div class="kpi"><span class="kpi__label">Score base</span><span class="kpi__value">{{ sens.baselineTotalScore | number:'1.2-2' }}</span></div>
                  <div class="kpi"><span class="kpi__label">Rosa</span><span class="kpi__value">{{ sens.baselineSquadSize }}</span></div>
                </div>
                @if (sens.warnings?.length) {
                  <ul class="warnings-list">@for (w of sens.warnings; track w) { <li>{{ w }}</li> }</ul>
                }
                @for (param of sens.parameters; track param.parameter) {
                  <div class="table-card">
                    <div class="table-card__head"><h3>{{ paramLabel(param.parameter) }}</h3></div>
                    <div class="table-scroll">
                      <table class="data-table">
                        <thead>
                          <tr>
                            <th>Valore</th><th>Status</th>
                            <th class="num">Score</th><th class="num">Δ</th><th class="num">Δ%</th>
                            <th class="num hide-sm">Jaccard</th><th class="num hide-sm">Δ rosa</th>
                          </tr>
                        </thead>
                        <tbody>
                          @for (pt of param.points; track pt.value) {
                            <tr [class.row-baseline]="pt.scoreDelta === 0">
                              <td>{{ pt.value | number:'1.2-2' }}</td>
                              <td><span class="status-chip">{{ pt.status }}</span></td>
                              <td class="num">{{ pt.totalScore | number:'1.2-2' }}</td>
                              <td class="num" [class.delta-pos]="pt.scoreDelta > 0" [class.delta-neg]="pt.scoreDelta < 0">{{ pt.scoreDelta | number:'1.2-2' }}</td>
                              <td class="num" [class.delta-pos]="pt.scoreDeltaPct > 0" [class.delta-neg]="pt.scoreDeltaPct < 0">{{ pt.scoreDeltaPct | number:'1.1-1' }}%</td>
                              <td class="num hide-sm">{{ pt.jaccardVsBaseline | number:'1.2-2' }}</td>
                              <td class="num hide-sm">{{ pt.playersChanged }}</td>
                            </tr>
                          }
                        </tbody>
                      </table>
                    </div>
                  </div>
                }
              } @else if (!analysisRunning()) {
                <div class="empty-state empty-state--compact">
                  <p class="empty-state__text">Avvia lo sweep per vedere come reagiscono score e composizione.</p>
                </div>
              }
            </div>
          }

          @if (analysisTab() === 'pareto') {
            <div class="analysis-body" role="tabpanel">
              <div class="analysis-actions">
                <button type="button" class="btn btn--secondary"
                        (click)="runPareto()" [disabled]="analysisRunning() || !canRun()">
                  @if (analysisRunning() && analysisTab() === 'pareto') {
                    <span class="spinner spinner--dark"></span> Calcolo…
                  } @else {
                    Esegui Pareto
                  }
                </button>
                <p class="muted">Trade-off score vs rischio e probabilità d’asta.</p>
              </div>
              @if (analysisError() && analysisTab() === 'pareto') {
                <app-error-boundary title="Errore Pareto" [message]="analysisError()!" />
              }
              @if (paretoResult(); as pr) {
                @if (pr.warnings?.length) {
                  <ul class="warnings-list">@for (w of pr.warnings; track w) { <li>{{ w }}</li> }</ul>
                }
                <div class="pareto-layout">
                  <div class="insight-card pareto-chart">
                    <svg class="pareto-svg" viewBox="0 0 320 220" role="img" aria-label="Scatter score vs risk">
                      <line x1="40" y1="10" x2="40" y2="190" class="pareto-axis" />
                      <line x1="40" y1="190" x2="310" y2="190" class="pareto-axis" />
                      <text x="8" y="20" class="pareto-axis-label">Score</text>
                      <text x="280" y="208" class="pareto-axis-label">Risk</text>
                      @for (pt of pr.points; track pt.riskLambda) {
                        <circle
                          [attr.cx]="paretoX(pt, pr.points)"
                          [attr.cy]="paretoY(pt, pr.points)"
                          [attr.r]="pt.dominated ? 4 : 6"
                          [class.pareto-dot--frontier]="!pt.dominated"
                          [class.pareto-dot--dominated]="pt.dominated" />
                      }
                    </svg>
                    <div class="pareto-legend">
                      <span><i class="swatch swatch--frontier"></i> Frontier</span>
                      <span><i class="swatch swatch--dom"></i> Dominated</span>
                    </div>
                  </div>
                  <div class="table-card">
                    <div class="table-scroll">
                      <table class="data-table">
                        <thead>
                          <tr>
                            <th>λ</th><th>Status</th>
                            <th class="num">Score</th><th class="num">Risk</th>
                            <th class="num hide-sm">P(asta)</th>
                            <th class="num hide-sm">Size</th>
                            <th>Front.</th>
                          </tr>
                        </thead>
                        <tbody>
                          @for (pt of pr.points; track pt.riskLambda) {
                            <tr [class.row-baseline]="!pt.dominated">
                              <td>{{ pt.riskLambda | number:'1.2-2' }}</td>
                              <td><span class="status-chip">{{ pt.status }}</span></td>
                              <td class="num">{{ pt.score | number:'1.2-2' }}</td>
                              <td class="num">{{ pt.risk | number:'1.2-2' }}</td>
                              <td class="num hide-sm">{{ pt.winProbability != null ? (pt.winProbability | percent:'1.0-0') : '—' }}</td>
                              <td class="num hide-sm">{{ pt.squadSize }}</td>
                              <td>{{ pt.dominated ? '—' : '✓' }}</td>
                            </tr>
                          }
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              } @else if (!analysisRunning()) {
                <div class="empty-state empty-state--compact">
                  <p class="empty-state__text">Calcola la frontiera per confrontare score, rischio e fattibilità d’asta.</p>
                </div>
              }
            </div>
          }
        </section>
      </div>

      <!-- Mobile sticky CTA -->
      <div class="opt-mobile-cta">
        <button type="button" class="btn btn--primary btn--run"
                (click)="run(); mobilePane.set('results')"
                [disabled]="running() || !canRun()">
          @if (running()) {
            <span class="spinner"></span> Calcolo…
          } @else {
            Ottimizza rosa
          }
        </button>
      </div>
    </div>

    @if (selectedPlayer(); as p) {
      <app-optimizer-player-drawer [player]="p" (closed)="selectedPlayer.set(null)" />
    }
  `,
  styles: [`
    :host { display: block; }

    .sr-only {
      position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
      overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
    }

    .optimizer-page {
      display: flex;
      flex-direction: column;
      min-height: 100%;
      background: var(--color-bg, #0b0c0f);
      color: var(--color-text-primary, #f4f4f5);
    }

    /* ── Top bar ─────────────────────────────────────── */
    .opt-topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--color-border, #27272a);
      background: color-mix(in srgb, var(--color-bg, #0b0c0f) 88%, transparent);
      backdrop-filter: blur(10px);
    }
    .opt-topbar__title {
      margin: 0;
      font-size: 1.125rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      line-height: 1.25;
    }
    .opt-topbar__subtitle {
      margin: 2px 0 0;
      font-size: 0.75rem;
      color: var(--color-text-secondary, #a1a1aa);
      line-height: 1.35;
      max-width: 36rem;
    }
    .opt-topbar__actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .opt-preset { min-width: 140px; max-width: 220px; flex: 1 1 140px; }
    .opt-preset-banner {
      margin: 0;
      padding: 8px 16px;
      font-size: 0.75rem;
      line-height: 1.4;
      color: var(--color-text-secondary, #a1a1aa);
      border-bottom: 1px solid var(--color-border, #27272a);
      background: var(--color-surface, #14151a);
    }
    .opt-preset-banner.muted { opacity: 0.85; }

    /* ── Mobile pane nav ─────────────────────────────── */
    .opt-pane-nav {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--color-border, #27272a);
      background: var(--color-surface, #14151a);
      position: sticky;
      top: 57px;
      z-index: 15;
    }
    @media (min-width: 960px) {
      .opt-pane-nav { display: none; }
    }
    .opt-pane-nav__btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-height: 40px;
      padding: 8px 6px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      color: var(--color-text-secondary, #a1a1aa);
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
    }
    .opt-pane-nav__btn.active {
      background: var(--color-bg, #0b0c0f);
      color: var(--color-text-primary, #f4f4f5);
      border-color: var(--color-border, #27272a);
    }
    .opt-pane-nav__badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      padding: 0 5px;
      border-radius: 999px;
      background: var(--color-accent, #6366f1);
      color: #fff;
      font-size: 0.65rem;
      font-weight: 700;
    }

    /* ── Shell layout ────────────────────────────────── */
    .opt-shell {
      display: grid;
      grid-template-columns: 1fr;
      gap: 0;
      flex: 1;
      min-height: 0;
    }
    @media (min-width: 960px) {
      .opt-shell {
        grid-template-columns: minmax(320px, 400px) minmax(0, 1fr);
        grid-template-rows: auto auto;
        align-items: start;
      }
      .opt-config {
        grid-row: 1 / span 2;
        grid-column: 1;
        position: sticky;
        top: 72px;
        max-height: calc(100dvh - 80px);
        overflow-y: auto;
        border-right: 1px solid var(--color-border, #27272a);
      }
      .opt-results { grid-column: 2; grid-row: 1; }
      .opt-analysis { grid-column: 2; grid-row: 2; }
    }
    @media (min-width: 1280px) {
      .opt-shell {
        grid-template-columns: minmax(360px, 420px) minmax(0, 1.2fr) minmax(280px, 0.9fr);
        grid-template-rows: 1fr;
      }
      .opt-config { grid-row: 1; grid-column: 1; }
      .opt-results { grid-column: 2; grid-row: 1; }
      .opt-analysis {
        grid-column: 3; grid-row: 1;
        position: sticky;
        top: 72px;
        max-height: calc(100dvh - 80px);
        overflow-y: auto;
        border-left: 1px solid var(--color-border, #27272a);
      }
    }

    .opt-pane {
      padding: 12px 16px 88px;
      min-width: 0;
    }
    @media (min-width: 960px) {
      .opt-pane { padding: 16px; display: block !important; }
      .opt-mobile-cta { display: none !important; }
    }
    @media (max-width: 959px) {
      .opt-pane { display: none; }
      .opt-pane--active { display: block; }
    }

    /* ── Accordion sections ──────────────────────────── */
    .opt-section {
      border: 1px solid var(--color-border, #27272a);
      border-radius: 12px;
      background: var(--color-surface, #14151a);
      margin-bottom: 10px;
      overflow: hidden;
    }
    .opt-section__head {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto auto;
      align-items: center;
      gap: 8px;
      padding: 12px 14px;
      min-height: 48px;
      border: none;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .opt-section__head:hover { background: color-mix(in srgb, var(--color-accent, #6366f1) 6%, transparent); }
    .opt-section__title { font-size: 0.875rem; font-weight: 650; }
    .opt-section__hint {
      font-size: 0.7rem;
      color: var(--color-text-secondary, #a1a1aa);
      justify-self: end;
    }
    .opt-section__chev {
      width: 8px; height: 8px;
      border-right: 2px solid var(--color-text-secondary, #a1a1aa);
      border-bottom: 2px solid var(--color-text-secondary, #a1a1aa);
      transform: rotate(45deg);
      transition: transform 150ms ease;
      margin-bottom: 4px;
    }
    .opt-section.open .opt-section__chev { transform: rotate(225deg); margin-bottom: 0; margin-top: 4px; }
    .opt-section__chev--open { transform: rotate(225deg); margin-bottom: 0; margin-top: 4px; }

    /* ── Advanced config toggle ──────────────────────── */
    /* Dashed border + muted surface (vs. .opt-section's solid border) marks
       this as a meta-toggle for a group, not another data-bearing section. */
    .opt-advanced {
      border: 1px dashed var(--color-border, #27272a);
      border-radius: 12px;
      margin-bottom: 10px;
      overflow: hidden;
    }
    .opt-advanced__toggle {
      width: 100%;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      min-height: 44px;
      border: none;
      background: transparent;
      color: var(--color-text-secondary, #a1a1aa);
      text-align: left;
      cursor: pointer;
    }
    .opt-advanced__toggle:hover { color: var(--color-text-primary, #f4f4f5); }
    .opt-advanced__title { font-size: 0.8125rem; font-weight: 600; }
    .opt-advanced__badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      padding: 0 5px;
      border-radius: 999px;
      background: var(--color-border, #27272a);
      color: var(--color-text-secondary, #a1a1aa);
      font-size: 0.65rem;
      font-weight: 700;
    }
    .opt-advanced__toggle .opt-section__chev { margin-left: auto; }
    .opt-advanced__hint {
      margin: 0;
      padding: 0 14px 12px;
      font-size: 0.7rem;
      line-height: 1.4;
      color: var(--color-text-secondary, #a1a1aa);
    }
    .opt-section__body {
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 0 14px 14px;
      border-top: 1px solid var(--color-border, #27272a);
      padding-top: 12px;
    }

    /* ── Form controls ───────────────────────────────── */
    .field-group { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
    .field-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    @media (min-width: 480px) {
      .field-row { grid-template-columns: 1fr 1fr; }
    }
    .field-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--color-text-primary, #f4f4f5);
    }
    .field-hint {
      font-weight: 500;
      color: var(--color-text-secondary, #a1a1aa);
      font-size: 0.7rem;
    }
    .field-input {
      width: 100%;
      min-height: 40px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-bg, #0b0c0f);
      color: var(--color-text-primary, #f4f4f5);
      font-size: 0.875rem;
    }
    .field-input--compact { min-height: 36px; font-size: 0.8125rem; }
    .field-input:focus {
      outline: 2px solid color-mix(in srgb, var(--color-accent, #6366f1) 55%, transparent);
      outline-offset: 1px;
      border-color: var(--color-accent, #6366f1);
    }
    .field-textarea { resize: vertical; min-height: 64px; font-family: inherit; }

    .chip-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 36px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-bg, #0b0c0f);
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--color-text-secondary, #a1a1aa);
      cursor: pointer;
      user-select: none;
    }
    .chip input { position: absolute; opacity: 0; pointer-events: none; }
    .chip.active {
      color: var(--color-text-primary, #f4f4f5);
      border-color: var(--color-accent, #6366f1);
      background: color-mix(in srgb, var(--color-accent, #6366f1) 12%, transparent);
    }
    .chip__icon { font-size: 0.85rem; }

    .toggle-row {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-bg, #0b0c0f);
      cursor: pointer;
      font-size: 0.8125rem;
      line-height: 1.35;
    }
    .toggle-row input { margin-top: 3px; width: 16px; height: 16px; accent-color: var(--color-accent, #6366f1); }
    .toggle-row strong { display: block; }
    .toggle-row .field-hint { display: block; margin-top: 2px; }

    .weight-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 8px;
    }
    .link-btn {
      align-self: flex-start;
      border: none;
      background: none;
      color: var(--color-accent, #6366f1);
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      padding: 4px 0;
      min-height: 32px;
    }

    /* ── Buttons ─────────────────────────────────────── */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 40px;
      padding: 8px 14px;
      border-radius: 10px;
      border: 1px solid transparent;
      font-size: 0.8125rem;
      font-weight: 650;
      cursor: pointer;
      transition: opacity 120ms, border-color 120ms, background 120ms;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn--primary {
      background: var(--color-accent, #6366f1);
      color: #fff;
    }
    .btn--primary:not(:disabled):hover { opacity: 0.92; }
    .btn--secondary {
      background: transparent;
      color: var(--color-text-primary, #f4f4f5);
      border-color: var(--color-border, #27272a);
    }
    .btn--secondary:not(:disabled):hover { border-color: var(--color-accent, #6366f1); }
    .btn--run { min-width: 140px; white-space: nowrap; }

    .spinner {
      width: 14px; height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }
    .spinner--dark {
      border-color: color-mix(in srgb, var(--color-text-primary, #fff) 25%, transparent);
      border-top-color: var(--color-text-primary, #fff);
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .inline-note {
      margin: 0;
      padding: 8px 10px;
      border-radius: 8px;
      background: color-mix(in srgb, var(--color-accent, #6366f1) 10%, transparent);
      border: 1px solid color-mix(in srgb, var(--color-accent, #6366f1) 25%, transparent);
      font-size: 0.75rem;
      color: var(--color-text-secondary, #a1a1aa);
    }

    /* ── Empty states ────────────────────────────────── */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 12px;
      padding: 32px 16px;
      min-height: 240px;
    }
    .empty-state--compact { min-height: 120px; padding: 20px 12px; }
    .empty-state__icon { font-size: 2rem; }
    .empty-state__title { margin: 0; font-size: 1rem; font-weight: 700; }
    .empty-state__text {
      margin: 0;
      max-width: 28rem;
      font-size: 0.8125rem;
      line-height: 1.45;
      color: var(--color-text-secondary, #a1a1aa);
    }
    .empty-state__stack {
      width: 100%;
      max-width: 420px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    /* ── Results ─────────────────────────────────────── */
    .strategy-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .strategy-tab {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 36px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-surface, #14151a);
      color: var(--color-text-secondary, #a1a1aa);
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
    }
    .strategy-tab.active {
      color: var(--color-text-primary, #f4f4f5);
      border-color: var(--color-accent, #6366f1);
      background: color-mix(in srgb, var(--color-accent, #6366f1) 12%, transparent);
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    @media (min-width: 640px) {
      .kpi-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
    .kpi-grid--compact { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .kpi {
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-surface, #14151a);
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }
    .kpi__label {
      font-size: 0.65rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--color-text-secondary, #a1a1aa);
    }
    .kpi__value {
      font-size: 1.25rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.02em;
    }
    .kpi__value--sm { font-size: 0.9rem; }
    .kpi__sub {
      font-size: 0.7rem;
      color: var(--color-text-secondary, #a1a1aa);
    }

    .role-strip-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
      margin-bottom: 12px;
    }
    .role-strip {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      padding: 8px 4px;
      border-radius: 10px;
      border: 1px solid var(--color-border, #27272a);
      border-top-width: 3px;
      background: var(--color-surface, #14151a);
    }
    .role-strip__label { font-size: 0.65rem; font-weight: 700; }
    .role-strip__count { font-size: 1rem; font-weight: 700; }

    .formation-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .formation-chip {
      font-size: 0.7rem;
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 6px;
      border: 1px solid var(--color-border, #27272a);
      color: var(--color-text-secondary, #a1a1aa);
    }
    .formation-chip.ok { color: #16a34a; border-color: color-mix(in srgb, #16a34a 40%, transparent); }
    .formation-chip.ko { color: #dc2626; border-color: color-mix(in srgb, #dc2626 40%, transparent); }

    .insight-card {
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-surface, #14151a);
      margin-bottom: 12px;
    }
    .insight-card h3, .table-card__head h3, .analysis-intro h2 {
      margin: 0;
      font-size: 0.875rem;
      font-weight: 650;
    }
    .insight-card__head {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 8px;
    }
    .near-item {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 0.8125rem;
      padding: 4px 0;
    }

    .table-card {
      border: 1px solid var(--color-border, #27272a);
      border-radius: 12px;
      background: var(--color-surface, #14151a);
      overflow: hidden;
      margin-bottom: 12px;
    }
    .table-card__head {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: baseline;
      gap: 6px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--color-border, #27272a);
    }
    .table-scroll { overflow: auto; -webkit-overflow-scrolling: touch; }
    .data-table {
      width: 100%;
      min-width: 480px;
      border-collapse: collapse;
      font-size: 0.8125rem;
    }
    .data-table thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--color-surface, #14151a);
      padding: 8px 10px;
      text-align: left;
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--color-text-secondary, #a1a1aa);
      border-bottom: 1px solid var(--color-border, #27272a);
    }
    .data-table tbody td {
      padding: 10px;
      border-bottom: 1px solid var(--color-border, #27272a);
      vertical-align: middle;
    }
    .data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
    .name-cell { font-weight: 560; }
    .row-click { cursor: pointer; }
    .row-click:hover, .row-click:focus-visible {
      background: color-mix(in srgb, var(--color-accent, #6366f1) 8%, transparent);
      outline: none;
    }
    .row-baseline { background: color-mix(in srgb, var(--color-accent, #6366f1) 7%, transparent); }
    .hide-sm { display: none; }
    @media (min-width: 640px) {
      .hide-sm { display: table-cell; }
      .data-table { min-width: 0; }
    }

    .role-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 36px;
      padding: 2px 6px;
      border-radius: 6px;
      border: 1px solid;
      font-size: 0.65rem;
      font-weight: 700;
    }
    .accent { color: var(--color-accent, #6366f1); font-weight: 650; }
    .muted { color: var(--color-text-secondary, #a1a1aa); font-size: 0.75rem; }
    .delta-pos { color: #16a34a; }
    .delta-neg { color: #dc2626; }
    .status-chip {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 6px;
      border: 1px solid var(--color-border, #27272a);
      font-size: 0.7rem;
      color: var(--color-text-secondary, #a1a1aa);
    }
    .warnings-list {
      margin: 8px 0 0;
      padding-left: 18px;
      font-size: 0.75rem;
      color: var(--color-text-secondary, #a1a1aa);
    }

    /* ── Analysis ────────────────────────────────────── */
    .analysis-intro { margin-bottom: 12px; }
    .analysis-intro p { margin: 6px 0 0; line-height: 1.45; }
    .analysis-tabs {
      display: inline-flex;
      gap: 4px;
      padding: 3px;
      border-radius: 10px;
      border: 1px solid var(--color-border, #27272a);
      background: var(--color-surface, #14151a);
      margin-bottom: 12px;
    }
    .analysis-tab {
      min-height: 34px;
      padding: 6px 12px;
      border: none;
      border-radius: 8px;
      background: transparent;
      color: var(--color-text-secondary, #a1a1aa);
      font-size: 0.75rem;
      font-weight: 650;
      cursor: pointer;
    }
    .analysis-tab.active {
      background: var(--color-bg, #0b0c0f);
      color: var(--color-text-primary, #f4f4f5);
      box-shadow: 0 0 0 1px var(--color-border, #27272a);
    }
    .analysis-body { display: flex; flex-direction: column; gap: 12px; }
    .analysis-actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }
    .analysis-actions p { margin: 0; max-width: 28rem; line-height: 1.4; }

    .pareto-layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    @media (min-width: 720px) {
      .opt-analysis .pareto-layout { grid-template-columns: minmax(240px, 320px) 1fr; }
    }
    .pareto-svg { width: 100%; height: auto; display: block; }
    .pareto-axis { stroke: var(--color-border, #27272a); stroke-width: 1; }
    .pareto-axis-label { fill: var(--color-text-secondary, #a1a1aa); font-size: 10px; }
    .pareto-dot--frontier { fill: var(--color-accent, #6366f1); }
    .pareto-dot--dominated { fill: var(--color-text-secondary, #a1a1aa); opacity: 0.45; }
    .pareto-legend {
      display: flex;
      gap: 12px;
      margin-top: 8px;
      font-size: 0.7rem;
      color: var(--color-text-secondary, #a1a1aa);
    }
    .pareto-legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .swatch--frontier { background: var(--color-accent, #6366f1); }
    .swatch--dom { background: var(--color-text-secondary, #a1a1aa); opacity: 0.45; }

    /* ── Mobile sticky CTA ───────────────────────────── */
    .opt-mobile-cta {
      position: sticky;
      bottom: 0;
      z-index: 20;
      padding: 10px 12px calc(10px + env(safe-area-inset-bottom));
      border-top: 1px solid var(--color-border, #27272a);
      background: color-mix(in srgb, var(--color-bg, #0b0c0f) 90%, transparent);
      backdrop-filter: blur(10px);
    }
    .opt-mobile-cta .btn { width: 100%; min-height: 44px; }
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
  readonly seasonStart = signal<number>(SEASON_FALLBACK_LIST[0]);
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
  /** Official Mantra module preferred (soft unless enforcePreferredMantra). */
  readonly preferredMantraFormation = signal<string>('');
  readonly enforcePreferredMantra = signal(false);
  readonly mantraModuleLabels = MANTRA_MODULE_LABELS;

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
  readonly hybridBlend = signal(0.0);
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

  // ── Analysis tools (Sensitivity / Pareto) ─────────────
  readonly analysisTab = signal<'sensitivity' | 'pareto'>('sensitivity');
  readonly analysisRunning = signal(false);
  readonly analysisError = signal<string | null>(null);
  readonly sensitivityResult = signal<SensitivityResponse | null>(null);
  readonly paretoResult = signal<ParetoResponse | null>(null);

  // ── UI chrome (responsive panes + config accordions) ──
  /** Mobile-only primary pane. On ≥960px all panes are visible. */
  readonly mobilePane = signal<'config' | 'results' | 'analysis'>('config');
  /** Open accordion sections in the config column. */
  readonly openSections = signal<ReadonlySet<string>>(new Set(['essentials', 'objective']));
  /** Whether the "advanced config" group (constraints/objective/robustness/
   * formations/filters) is expanded. Collapsed by default: these are expert-
   * level knobs (VAR blend, ESV weight, Monte Carlo, inflation curve, ...)
   * that shouldn't compete visually with the essentials on first load. */
  readonly showAdvancedConfig = signal(false);

  isSectionOpen(id: string): boolean {
    return this.openSections().has(id);
  }

  toggleSection(id: string): void {
    this.openSections.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

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
        this.seasons.set([...SEASON_FALLBACK_LIST]);
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
    if (req.hybridBlend != null) this.hybridBlend.set(req.hybridBlend);
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
    if (req.preferredMantraFormation) {
      this.preferredMantraFormation.set(req.preferredMantraFormation);
    } else if (req.preferredMantraFormation === null) {
      this.preferredMantraFormation.set('');
    }
    if (typeof req.enforcePreferredMantraFormation === 'boolean') {
      this.enforcePreferredMantra.set(req.enforcePreferredMantraFormation);
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
        this.mobilePane.set('results');
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
            this.mobilePane.set('results');
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
      preferredMantraFormation:
        this.ruleset() === 'MANTRA' && this.preferredMantraFormation()
          ? this.preferredMantraFormation()
          : null,
      enforcePreferredMantraFormation:
        this.ruleset() === 'MANTRA' && this.enforcePreferredMantra(),
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
      hybridBlend: this.hybridBlend(),
      esvWeight: this.esvWeight(),
      valuationMode: this.valuationMode(),
      minStartProbability: this.minStartProbability(),
      replacementMethod: this.replacementMethod(),
      strategyNames: this.showCustomWeights() ? null : [...this.selectedStrategies()],
      customStrategies: this.showCustomWeights() ? this._buildCustomStrategies() : null,
    };
  }

  /**
   * Analysis endpoints require exactly one strategy. Prefer the single selected
   * strategy, otherwise fall back to BALANCED (or the first selected).
   */
  private _buildAnalysisRequest(): OptimizationRequest {
    const base = this._buildRequest();
    const name = this._resolveStrategyName(base);
    if (this.showCustomWeights()) {
      const customs = this._buildCustomStrategies();
      return {
        ...base,
        strategyNames: null,
        customStrategies: customs.length ? [customs[0]] : [{
          name,
          roleWeight: { ...this.customWeights() },
          minBudgetShareByRoles: null,
          maxTopTierPlayers: null,
          topTierCostThreshold: null,
        }],
        // Sensitivity/Pareto are deterministic sweeps — drop MC noise.
        monteCarlo: undefined,
        nearOptimal: undefined,
      };
    }
    return {
      ...base,
      strategyNames: [name],
      customStrategies: null,
      monteCarlo: undefined,
      nearOptimal: undefined,
    };
  }

  runSensitivity(): void {
    if (!this.canRun() || this.analysisRunning()) return;
    this.analysisRunning.set(true);
    this.analysisError.set(null);
    const req = this._buildAnalysisRequest();
    this.optimizerService.runSensitivity(req).pipe(
      takeUntilDestroyed(this.destroyRef),
      catchError(err => {
        const detail = err?.error?.detail ?? err?.message ?? 'Sensitivity request failed';
        this.analysisError.set(typeof detail === 'string' ? detail : JSON.stringify(detail));
        return of(null);
      }),
      finalize(() => this.analysisRunning.set(false)),
    ).subscribe(res => {
      if (res) this.sensitivityResult.set(res);
    });
  }

  runPareto(): void {
    if (!this.canRun() || this.analysisRunning()) return;
    this.analysisRunning.set(true);
    this.analysisError.set(null);
    const req = this._buildAnalysisRequest();
    this.optimizerService.runPareto(req).pipe(
      takeUntilDestroyed(this.destroyRef),
      catchError(err => {
        const detail = err?.error?.detail ?? err?.message ?? 'Pareto request failed';
        this.analysisError.set(typeof detail === 'string' ? detail : JSON.stringify(detail));
        return of(null);
      }),
      finalize(() => this.analysisRunning.set(false)),
    ).subscribe(res => {
      if (res) this.paretoResult.set(res);
    });
  }

  paramLabel(parameter: string): string {
    const map: Record<string, string> = {
      risk_aversion: 'Risk aversion',
      var_blend: 'VAR blend',
      hybrid_blend: 'Hybrid blend',
      budget: 'Budget',
    };
    return map[parameter] ?? parameter;
  }

  /** Map Δ% into a 0–100 bar width (capped). */
  deltaBarWidth(pct: number): number {
    const abs = Math.min(Math.abs(pct), 25);
    return (abs / 25) * 100;
  }

  paretoX(pt: ParetoPoint, points: ParetoPoint[]): number {
    const risks = points.map(p => p.risk);
    const min = Math.min(...risks);
    const max = Math.max(...risks);
    const span = max - min || 1;
    return 40 + ((pt.risk - min) / span) * 260;
  }

  paretoY(pt: ParetoPoint, points: ParetoPoint[]): number {
    const scores = points.map(p => p.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const span = max - min || 1;
    // SVG y grows downward
    return 190 - ((pt.score - min) / span) * 170;
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
    return Object.entries(r.formationFeasibility ?? {});
  }

  mantraFormationEntries(r: OptimizationResult): [string, boolean][] {
    const m = r.mantraFormationFeasibility;
    if (!m) return [];
    return Object.entries(m).map(([label, cov]) => [label, cov.feasible]);
  }

  mantraDeficitTitle(r: OptimizationResult, label: string): string {
    const cov = r.mantraFormationFeasibility?.[label];
    if (!cov) return label;
    if (cov.feasible) return `${label}: schierabile`;
    const parts = Object.entries(cov.deficits ?? {}).map(([k, v]) => `${k} −${v}`);
    return parts.length ? `${label}: manca ${parts.join(', ')}` : `${label}: non schierabile`;
  }

  meta(name: string) {
    return STRATEGY_META[name] ?? { label: name, icon: '📋' };
  }

  roleColor(role: string): string { return ROLE_COLORS[role] ?? 'var(--color-text-secondary)'; }
  roleLabel(role: string): string { return ROLE_LABELS[role] ?? role; }
}
