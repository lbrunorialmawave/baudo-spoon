import {
  Component, computed, inject, signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { OptimizerService } from '../../core/services/optimizer.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  FormationConfig,
  MultiStrategyResult,
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

const ALL_FORMATIONS: FormationConfig[] = [
  { label: '3-4-3', defenders: 3, midfielders: 4, forwards: 3 },
  { label: '4-3-3', defenders: 4, midfielders: 3, forwards: 3 },
  { label: '4-4-2', defenders: 4, midfielders: 4, forwards: 2 },
  { label: '3-5-2', defenders: 3, midfielders: 5, forwards: 2 },
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
    description: 'Anno di inizio della stagione di Serie A da cui pescare le quotazioni storiche, i listini e le statistiche dei giocatori. Il risolutore ILP usa queste informazioni per proiettare i punteggi attesi di fine stagione.',
    examples: [
      { label: '2025', value: 'stagione corrente 2025/26 (default)' },
      { label: '2024', value: 'stagione precedente 2024/25' },
    ],
  },
  budget: {
    description: 'Crediti totali (cr.) a disposizione per costruire l\'intera rosa. Il risolutore ILP cercherà di non superare questo budget, distribuiti tra tutti i ruoli secondo i vincoli impostati.',
    examples: [
      { label: '300 cr.', value: 'lega FantaSanremo classica' },
      { label: '500 cr.', value: 'lega con top player costosi' },
      { label: '1000 cr.', value: 'lega premium / fanta-manageriale' },
    ],
  },
  numParticipants: {
    description: 'Numero di squadre che partecipano alla lega. Influenza il modello di inflazione dei prezzi: più partecipanti significano maggiore competizione e prezzi attesi più alti.',
    examples: [
      { label: '4', value: 'lega piccola' },
      { label: '8', value: 'default classico' },
      { label: '10–12', value: 'lega grande' },
    ],
  },
  minQtA: {
    description: 'Quota minima d\'asta per ruolo (Minimum Quota per Auction): imposta un tetto minimo di crediti da spendere per ciascun ruolo, per evitare rose sbilanciate. 0 = nessun vincolo.',
    examples: [
      { label: '0', value: 'nessun vincolo, libera distribuzione' },
      { label: '1', value: 'almeno 1 cr. investito per ciascun ruolo' },
      { label: '3', value: 'vincolo più stringente, es. minimo 3 P, 3 D, ecc.' },
    ],
  },
  solverTimeoutSeconds: {
    description: 'Tempo massimo (in secondi) concesso al risolutore ILP per trovare la soluzione ottima. Allo scadere, restituirà la migliore soluzione ammissibile trovata finora (se disponibile).',
    examples: [
      { label: '10 s', value: 'run veloce, utile per test' },
      { label: '30 s', value: 'default bilanciato' },
      { label: '120 s', value: 'run approfondito per leghe complesse' },
    ],
  },
  minDistinctTeams: {
    description: 'Numero minimo di squadre di Serie A diverse da cui devono provenire i giocatori della tua rosa. Garantisce diversificazione e riduce il rischio di concentrazione su poche squadre.',
    examples: [
      { label: '4', value: 'minimo sindacale, pochi vincoli' },
      { label: '12', value: 'default, ben diversificato' },
      { label: '18', value: 'massima diversificazione' },
    ],
  },
  maxPlayersPerTeam: {
    description: 'Numero massimo di giocatori che puoi avere dalla stessa squadra di Serie A. Impedisce di costruire una rosa troppo dipendente da un singolo club.',
    examples: [
      { label: '2', value: 'vincolo stretto' },
      { label: '4', value: 'default' },
      { label: '6', value: 'consentita alta concentrazione' },
    ],
  },
  bigTeamsCap: {
    description: 'Numero massimo di giocatori acquistabili dalle "big team" (elencate sotto). Insieme al vincolo max-players-per-team, impedisce di costruire una rosa di soli top club.',
    examples: [
      { label: '3', value: 'default: massimo 3 giocatori dalle big team' },
      { label: '0', value: 'nessun giocatore dalle big team (lega anti-big)' },
      { label: '10', value: 'nessun vincolo aggiuntivo oltre max-per-team' },
    ],
  },
  bigTeams: {
    description: 'Elenco, separato da virgole, delle squadre di Serie A considerate "big" ai fini del vincolo `bigTeamsCap`. I nomi devono corrispondere esattamente a quelli usati dalle statistiche (es. "Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta").',
    examples: [
      { label: 'Inter, Milan, Juventus, Napoli', value: 'default: top 4 tradizionali' },
      { label: 'Inter, Milan, Juve, Napoli, Roma, Lazio, Atalanta', value: 'top 7 allargato' },
      { label: '(vuoto)', value: 'nessuna big team, solo max-per-team' },
    ],
  },
  maxSinglePlayerBudgetShare: {
    description: 'Frazione massima del budget totale (0–1) che può essere spesa per un singolo giocatore. Evita che un top player "mangi" troppo budget e renda impossibile completare la rosa.',
    examples: [
      { label: '0.15', value: '15% del budget (es. 45 cr. su 300), rosa molto equa' },
      { label: '0.30', value: 'default, equilibrio fra top player e rosa' },
      { label: '0.50', value: 'ammessi top player da 150 cr. su 300' },
    ],
  },
  mustInclude: {
    description: 'Elenco di ID giocatore (formato `fm-XXXXX`, separati da virgola) che DEVONO essere nella rosa. Vincolo hard: il risolutore fallirà se non riesce a includerli.',
    examples: [
      { label: 'fm-12345', value: 'forza inclusione di un giocatore specifico' },
      { label: '(vuoto)', value: 'nessun vincolo' },
    ],
  },
  exclude: {
    description: 'Elenco di ID giocatore (formato `fm-XXXXX`, separati da virgola) che NON devono essere nella rosa. Utile per escludere infortunati, squalificati o giocatori che non vuoi.',
    examples: [
      { label: 'fm-67890', value: 'escludi un singolo giocatore' },
      { label: 'fm-1, fm-2, fm-3', value: 'escludi più giocatori' },
    ],
  },
  ruleset: {
    description: 'Regolamento della lega. "Classic" usa le regole tradizionali del Fantacalcio (3 P, 8 D, 8 C, 6 A); "Mantra" supporta ruoli aggiuntivi (Trequartista, Mediano, ecc.) e conversioni ruolo.',
    examples: [
      { label: 'CLASSIC', value: 'regolamento tradizionale italiano' },
      { label: 'MANTRA', value: 'regolamento Mantra con ruoli modulari' },
    ],
  },
  riskAversion: {
    description: 'Coefficiente (0–5) di avversione al rischio. 0 = neutrale (massimizza solo lo score atteso). Valori alti penalizzano soluzioni con score molto variabile, a favore di rose più "sicure" e costanti.',
    examples: [
      { label: '0.0', value: 'neutrale, solo performance attesa' },
      { label: '1.0', value: 'lieve penalizzazione per la varianza' },
      { label: '3.0+', value: 'forte avversione, rose conservative' },
    ],
  },
  varBlend: {
    description: 'Peso (0–1) del modello VAR (Value Above Replacement) nella funzione obiettivo. Un valore più alto spinge il risolutore a preferire giocatori il cui contributo è superiore a quello di un sostituto medio.',
    examples: [
      { label: '0.0', value: 'VAR non usato' },
      { label: '0.5', value: 'mix 50% score, 50% VAR' },
      { label: '1.0', value: 'ottimizzazione puramente VAR' },
    ],
  },
  esvWeight: {
    description: 'Peso (0–5) dell\'ESV (Expected Season Value) nella funzione obiettivo. Più alto = il risolutore premia giocatori con alto valore di stagione atteso rispetto al prezzo di acquisto.',
    examples: [
      { label: '0', value: 'ESV non considerato' },
      { label: '1', value: 'peso moderato, focus sui "best value"' },
      { label: '2+', value: 'forte peso, ricerca aggressiva di affari' },
    ],
  },
  valuationMode: {
    description: 'Metrica di valutazione con cui il risolutore ILP stima il "valore" di ciascun giocatore. Cambia la base usata dal modello VAR (Value Above Replacement) per ordinare i candidati in rosa.',
    examples: [
      { label: 'PER_MATCH_RATING', value: 'default: media di rendimento a partita (più stabile, premia i titolari)' },
      { label: 'SEASON_VALUE',      value: 'totale di stagione proiettato (più "pessimistico" se il giocatore salta gare)' },
    ],
  },
  replacementMethod: {
    description: 'Metodo per stimare il "replacement level", ovvero il valore del giocatore medio facilmente reperibile sul mercato. È il livello di confronto usato dal modello VAR: un giocatore vale X solo se il suo contributo supera il rimpiazzo di almeno X − replacement.',
    examples: [
      { label: 'percentile',   value: 'default: 10° percentile della distribuzione (sostituto "ragionevolmente scarso")' },
      { label: 'roster_depth', value: 'usa la profondità media delle rose avversarie, più reale in leghe con molte squadre' },
    ],
  },
  minStartProbability: {
    description: 'Probabilità minima (0–1) che un giocatore sia schierato titolare dal proprio club. Sotto questa soglia il giocatore viene filtrato e NON entra in rosa, perché il risolutore non può garantire rendimento su gare che non gioca. Vuoto = nessun filtro.',
    examples: [
      { label: 'null / vuoto', value: 'nessun filtro, include anche riserve e infortunati' },
      { label: '0.3',          value: 'esclude solo chi è praticamente fuori rosa' },
      { label: '0.7',          value: 'default consigliato, solo giocatori con alta titolarità' },
      { label: '0.9',          value: 'rosa di soli "intoccabili", scelte molto ridotte' },
    ],
  },
  formations: {
    description: 'Moduli tattici ammessi dal regolamento della tua lega. Il risolutore ILP verificherà la fattibilità per ciascuno e ti mostrerà quali sono fattibili (✓) e quali no (✗) per la rosa risultante.',
    examples: [
      { label: '3-4-3', value: 'molto offensivo, 3 attaccanti titolari' },
      { label: '4-4-2', value: 'classico italiano, equilibrato' },
      { label: '3-5-2', value: 'centrocampo folto, fase difensiva forte' },
    ],
  },
  preferredFormation: {
    description: 'Modulo tattico preferito come VINCOLO HARD. Se impostato (diverso da "Nessuna"), il risolutore restituirà solo soluzioni giocabili in questo modulo. Se nessuna soluzione è compatibile, la run fallirà.',
    examples: [
      { label: '3-5-2', value: 'forza il modulo a 3-5-2' },
      { label: 'Nessuna', value: 'nessun vincolo, sceglie tra tutte le formazioni selezionate sopra' },
    ],
  },
  inflationPercentileThreshold: {
    description: 'Soglia (0–1) di percentile usata dal modello di inflazione. I giocatori con quotazione sopra questo percentile sono considerati "rari" e subiscono un moltiplicatore maggiore.',
    examples: [
      { label: '0.5', value: 'default, top 50% subisce inflazione' },
      { label: '0.8', value: 'solo top 20% subisce inflazione forte' },
    ],
  },
  maxInflationMultiplier: {
    description: 'Moltiplicatore massimo (1–5) che il modello di inflazione può applicare al listino base. Un valore di 2.0 significa che un top player può costare fino al doppio del listino.',
    examples: [
      { label: '1.0', value: 'nessuna inflazione, prezzi = listino' },
      { label: '1.6', value: 'default moderato' },
      { label: '3.0', value: 'lega molto calda, top player esplosivi' },
    ],
  },
  baseInflationRate: {
    description: 'Tasso base di inflazione (0–1) applicato a tutti i prezzi anche al di sotto della soglia di percentile. Compensa l\'effetto generale di "rialzo" del mercato.',
    examples: [
      { label: '0.00', value: 'nessun rialzo base' },
      { label: '0.05', value: 'default, +5% su tutti i listini' },
      { label: '0.15', value: 'mercato molto rialzista' },
    ],
  },
  baselineParticipants: {
    description: 'Numero di partecipanti "baseline" usato dal modello di inflazione per calcolare la pressione di mercato. Tipicamente pari al valore del campo Partecipanti; modifica solo se vuoi simulare una lega di dimensione diversa.',
    examples: [
      { label: '4', value: 'simula mercato di lega piccola' },
      { label: '8', value: 'default, mercato standard' },
      { label: '12', value: 'simula mercato affollato' },
    ],
  },
  teamStrengthMultiplier: {
    description: 'Peso dell\'aggiustamento Elo di Club sulla stima del costo dei giocatori. 0 = disattivato (default backend), valori più alti premiano i giocatori di squadre con Elo alto nel prezzo stimato (più costosi per le big, meno costosi per le piccole). Il moltiplicatore agisce solo sull\'effective cost usato dal risolutore, non sul listino secco.',
    examples: [
      { label: '0.0', value: 'disattivato, Elo ignorato' },
      { label: '0.5', value: 'aggiustamento moderato, leggero premio alle big' },
      { label: '1.0', value: 'aggiustamento pieno, differenza di costo tra big e piccole marcata' },
    ],
  },
  customWeights: {
    description: 'Pesi per ruolo (P/D/C/A) usati SOLO quando è selezionata una singola strategia e si sceglie di personalizzarla. Un peso più alto spinge il risolutore a investire di più in quel ruolo rispetto al bilanciamento standard.',
    examples: [
      { label: 'P=1 D=1 C=1 A=1', value: 'bilanciamento neutro (default)' },
      { label: 'P=0.5 D=2.5 C=1 A=1', value: 'focus difensori, rosa a basso rischio' },
      { label: 'P=1 D=1 C=1 A=2.5', value: 'top player offensivi' },
    ],
  },
};

@Component({
  selector: 'app-optimizer',
  standalone: true,
  imports: [FormsModule, DecimalPipe, PercentPipe, SkeletonComponent, ErrorBoundaryComponent, FieldLegendComponent],
  template: `
    <div class="optimizer-page">

      <header class="page-header">
        <div>
          <h1 class="page-title">Ottimizzatore Rosa</h1>
          <p class="page-subtitle">Costruzione automatica della rosa tramite risolutore ILP · 4 strategie disponibili</p>
        </div>
      </header>

      <div class="optimizer-body">

        <!-- ── Config panel ──────────────────────────────── -->
        <aside class="config-panel card">

          <!-- PRESETS -->
          <p class="section-divider">Profilo strategico</p>

          <div class="field-group">
            <label class="field-label" for="opt-preset">Preset di ottimizzazione</label>
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
          <p class="section-divider">Impostazioni di base</p>

          <div class="field-group">
            <label class="field-label" for="opt-seasonStart">Stagione di riferimento (Serie A)</label>
            <select id="opt-seasonStart" class="field-input" [(ngModel)]="seasonStart"
                    [attr.aria-describedby]="'legend-seasonStart'">
              @for (s of seasons(); track s) {
                <option [value]="s">{{ s }}/{{ s + 1 }}</option>
              }
            </select>
            <app-field-legend
              fieldId="legend-seasonStart"
              [description]="OPTIMIZER_LEGENDS['seasonStart'].description"
              [examples]="OPTIMIZER_LEGENDS['seasonStart'].examples" />
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-budget">Budget totale disponibile <span class="field-hint">cr.</span></label>
              <input id="opt-budget" class="field-input" type="number" min="200" max="1000" step="25"
                     [(ngModel)]="budget"
                     [attr.aria-describedby]="'legend-budget'" />
              <app-field-legend
                fieldId="legend-budget"
                [description]="OPTIMIZER_LEGENDS['budget'].description"
                [examples]="OPTIMIZER_LEGENDS['budget'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-numParticipants">Numero di partecipanti alla lega</label>
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
              <label class="field-label" for="opt-minQtA">Quota minima d'asta per ciascun ruolo</label>
              <input id="opt-minQtA" class="field-input" type="number" min="0" max="10" step="1"
                     [(ngModel)]="minQtA"
                     [attr.aria-describedby]="'legend-minQtA'" />
              <app-field-legend
                fieldId="legend-minQtA"
                [description]="OPTIMIZER_LEGENDS['minQtA'].description"
                [examples]="OPTIMIZER_LEGENDS['minQtA'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-solverTimeout">Timeout risolutore ILP <span class="field-hint">secondi</span></label>
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
          <p class="section-divider">Vincoli sulla rosa</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-minDistinctTeams">Numero minimo di squadre di Serie A distinte in rosa</label>
              <input id="opt-minDistinctTeams" class="field-input" type="number" min="1" max="25" step="1"
                     [(ngModel)]="minDistinctTeams"
                     [attr.aria-describedby]="'legend-minDistinctTeams'" />
              <app-field-legend
                fieldId="legend-minDistinctTeams"
                [description]="OPTIMIZER_LEGENDS['minDistinctTeams'].description"
                [examples]="OPTIMIZER_LEGENDS['minDistinctTeams'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-maxPlayersPerTeam">Numero massimo di giocatori per singola squadra</label>
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
              <label class="field-label" for="opt-bigTeamsCap">Limite totale di giocatori dalle "big team"</label>
              <input id="opt-bigTeamsCap" class="field-input" type="number" min="0" max="25" step="1"
                     [(ngModel)]="bigTeamsCap"
                     [attr.aria-describedby]="'legend-bigTeamsCap'" />
              <app-field-legend
                fieldId="legend-bigTeamsCap"
                [description]="OPTIMIZER_LEGENDS['bigTeamsCap'].description"
                [examples]="OPTIMIZER_LEGENDS['bigTeamsCap'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-maxShare">Quota massima di budget per singolo giocatore <span class="field-hint">0–1</span></label>
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
            <label class="field-label" for="opt-bigTeamsRaw">Elenco delle "big team" <span class="field-hint">separate da virgola</span></label>
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
          <p class="section-divider">Filtri sui giocatori</p>

          <div class="field-group">
            <label class="field-label" for="opt-mustInclude">Giocatori da includere obbligatoriamente <span class="field-hint">ID separati da virgola</span></label>
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
            <label class="field-label" for="opt-exclude">Giocatori da escludere <span class="field-hint">ID separati da virgola</span></label>
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
          <p class="section-divider">Regolamento e gestione del rischio</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-ruleset">Regolamento della lega</label>
              <select id="opt-ruleset" class="field-input" [(ngModel)]="ruleset"
                      [attr.aria-describedby]="'legend-ruleset'">
                <option value="CLASSIC">Classic (Fantacalcio tradizionale)</option>
                <option value="MANTRA">Mantra (ruoli modulari e conversioni)</option>
              </select>
              <app-field-legend
                fieldId="legend-ruleset"
                [description]="OPTIMIZER_LEGENDS['ruleset'].description"
                [examples]="OPTIMIZER_LEGENDS['ruleset'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-riskAversion">Avversione al rischio <span class="field-hint">0 = neutrale</span></label>
              <input id="opt-riskAversion" class="field-input" type="number" min="0" max="5" step="0.1"
                     [(ngModel)]="riskAversion"
                     [attr.aria-describedby]="'legend-riskAversion'" />
              <app-field-legend
                fieldId="legend-riskAversion"
                [description]="OPTIMIZER_LEGENDS['riskAversion'].description"
                [examples]="OPTIMIZER_LEGENDS['riskAversion'].examples" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-varBlend">Peso del modello VAR (Value Above Replacement) <span class="field-hint">0–1</span></label>
              <input id="opt-varBlend" class="field-input" type="number" min="0" max="1" step="0.1"
                     [(ngModel)]="varBlend"
                     [attr.aria-describedby]="'legend-varBlend'" />
              <app-field-legend
                fieldId="legend-varBlend"
                [description]="OPTIMIZER_LEGENDS['varBlend'].description"
                [examples]="OPTIMIZER_LEGENDS['varBlend'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-esvWeight">Peso dell'ESV (Expected Season Value) <span class="field-hint">0 = disattivato</span></label>
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
              <label class="field-label" for="opt-valuationMode">Metrica di valutazione</label>
              <select id="opt-valuationMode" class="field-input" [(ngModel)]="valuationMode"
                      [attr.aria-describedby]="'legend-valuationMode'">
                <option value="PER_MATCH_RATING">Per-match rating (default)</option>
                <option value="SEASON_VALUE">Season value (totale stagione)</option>
              </select>
              <app-field-legend
                fieldId="legend-valuationMode"
                [description]="OPTIMIZER_LEGENDS['valuationMode'].description"
                [examples]="OPTIMIZER_LEGENDS['valuationMode'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-replacementMethod">Metodo replacement level</label>
              <select id="opt-replacementMethod" class="field-input" [(ngModel)]="replacementMethod"
                      [attr.aria-describedby]="'legend-replacementMethod'">
                <option value="percentile">Percentile (10° pctile)</option>
                <option value="roster_depth">Roster depth</option>
              </select>
              <app-field-legend
                fieldId="legend-replacementMethod"
                [description]="OPTIMIZER_LEGENDS['replacementMethod'].description"
                [examples]="OPTIMIZER_LEGENDS['replacementMethod'].examples" />
            </div>
          </div>

          <div class="field-group">
            <label class="field-label" for="opt-minStartProb">Min. probabilità titolare <span class="field-hint">0–1, vuoto = nessun filtro</span></label>
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
          <p class="section-divider">Moduli tattici</p>

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
            <label class="field-label" for="opt-preferredFormation">Formazione preferita <span class="field-hint">vincolo hard</span></label>
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
          <p class="section-divider">Modello di inflazione dei prezzi</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label" for="opt-inflationPercentile">Soglia percentile di inflazione</label>
              <input id="opt-inflationPercentile" class="field-input" type="number" min="0" max="1" step="0.05"
                     [(ngModel)]="inflationPercentileThreshold"
                     [attr.aria-describedby]="'legend-inflationPercentileThreshold'" />
              <app-field-legend
                fieldId="legend-inflationPercentileThreshold"
                [description]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].description"
                [examples]="OPTIMIZER_LEGENDS['inflationPercentileThreshold'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-maxInflation">Moltiplicatore massimo di inflazione</label>
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
              <label class="field-label" for="opt-baseRate">Tasso base di inflazione</label>
              <input id="opt-baseRate" class="field-input" type="number" min="0" max="1" step="0.01"
                     [(ngModel)]="baseInflationRate"
                     [attr.aria-describedby]="'legend-baseInflationRate'" />
              <app-field-legend
                fieldId="legend-baseInflationRate"
                [description]="OPTIMIZER_LEGENDS['baseInflationRate'].description"
                [examples]="OPTIMIZER_LEGENDS['baseInflationRate'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-baselinePart">Partecipanti baseline per il modello</label>
              <input id="opt-baselinePart" class="field-input" type="number" min="2" max="20" step="1"
                     [(ngModel)]="baselineParticipants"
                     [attr.aria-describedby]="'legend-baselineParticipants'" />
              <app-field-legend
                fieldId="legend-baselineParticipants"
                [description]="OPTIMIZER_LEGENDS['baselineParticipants'].description"
                [examples]="OPTIMIZER_LEGENDS['baselineParticipants'].examples" />
            </div>
            <div class="field-group">
              <label class="field-label" for="opt-teamStrengthMul">Moltiplicatore Elo di Club (peso sulla stima del costo)</label>
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
          <p class="section-divider">Strategie di ottimizzazione</p>

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
              <span class="spinner"></span> Ottimizzazione in corso…
            } @else {
              Esegui ottimizzatore
            }
          </button>

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
                      <tr>
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
  `],
})
export class OptimizerComponent {
  private readonly optimizerService = inject(OptimizerService);
  private readonly quotationService = inject(QuotationService);

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
  readonly selectedPresetId = signal<string>(OPTIMIZER_PRESET_NONE);
  readonly activePreset = computed(() => findOptimizerPreset(this.selectedPresetId()));

  // Strategies loaded from API; fallback to known names if unavailable
  readonly availableStrategies = signal<string[]>(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']);

  // ── Basic ─────────────────────────────────────────────
  readonly seasons = signal<number[]>([]);
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
  readonly error = signal<string | null>(null);
  readonly results = signal<MultiStrategyResult | null>(null);
  readonly activeStrategy = signal<string>('');

  readonly resultKeys = computed(() => Object.keys(this.results()?.results ?? {}));
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
      },
      error: () => { this.seasons.set([2024, 2023, 2022]); },
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

    const bigTeams = this.bigTeamsRaw()
      .split(',').map(t => t.trim()).filter(Boolean);

    const formations = ALL_FORMATIONS.filter(f => this.selectedFormations().has(f.label));

    const mustInclude = this.mustIncludeRaw()
      .split(',').map(s => s.trim()).filter(Boolean);

    const exclude = this.excludeRaw()
      .split(',').map(s => s.trim()).filter(Boolean);

    const preferredLabel = this.preferredFormationLabel();
    const preferredFormation = preferredLabel
      ? (ALL_FORMATIONS.find(f => f.label === preferredLabel) ?? null)
      : null;

    this.optimizerService.runMulti({
      seasonStart: this.seasonStart(),
      budget: this.budget(),
      numParticipants: this.numParticipants(),
      minQtA: this.minQtA(),
      solverTimeoutSeconds: this.solverTimeoutSeconds(),
      minDistinctTeams: this.minDistinctTeams(),
      maxPlayersPerTeam: this.maxPlayersPerTeam(),
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
      varBlend: this.varBlend(),
      esvWeight: this.esvWeight(),
      valuationMode: this.valuationMode(),
      minStartProbability: this.minStartProbability(),
      replacementMethod: this.replacementMethod(),
      strategyNames: this.showCustomWeights() ? null : [...this.selectedStrategies()],
      customStrategies: this.showCustomWeights() ? this._buildCustomStrategies() : null,
    }).subscribe({
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

  setCustomWeight(role: string, value: number): void {
    this.customWeights.update(w => ({ ...w, [role]: +value }));
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
