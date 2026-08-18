import {
  Component,
  computed,
  effect,
  inject,
  input,
  signal,
} from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  AUCTION_PRESETS,
  AuctionPreset,
  AuctionPresetPolicy,
  findAuctionPreset,
} from '../../../core/constants/auction-presets';
import {
  AuctionConfig,
  AuctionParticipantSetup,
  AuctionSimulationResponse,
  BidderPolicy,
  BidderProfile,
  ParticipantSimStats,
  SimulateAuctionRequest,
} from '../../../core/models/auction.models';
import { AuctionService } from '../../../core/services/auction.service';

const DEFAULT_BOT_PRESET_IDS = ['balanced', 'aggressive', 'value_hunter'] as const;

/** Plain-language legend for metrics on each team card. */
const METRIC_HELP = {
  completion:
    'In quanti scenari questa squadra ha riempito tutti gli slot della rosa. 100% = roster completo in ogni run.',
  spend:
    'Crediti spesi a fine asta. p10 / p50 / p90: in 1 caso su 10 spendi meno di p10, in metà circa p50, in 1 su 10 più di p90.',
  esv:
    'Expected Surplus Value della rosa: valore in più rispetto al prezzo pagato. Più alto = rosa più conveniente sul mercato simulato.',
} as const;

function policyFromPreset(policy: AuctionPresetPolicy | undefined): BidderPolicy {
  if (!policy) return {};
  return {
    aggressiveness: policy.aggressiveness,
    inflationTolerance: policy.inflationTolerance,
    maxOverpayRatio: policy.maxOverpayRatio,
    minResidualCreditsPerSlot: policy.minResidualCreditsPerSlot,
    allInProbability: policy.allInProbability,
    budgetElasticity: policy.budgetElasticity,
    varWeight: policy.varWeight,
    teamStrengthWeight: policy.teamStrengthWeight,
    preferAlternatives: policy.preferAlternatives,
    preferLowCostAlternative: policy.preferLowCostAlternative,
    rebidTriggerPctAboveExpected: policy.rebidTriggerPctAboveExpected,
    budgetShareByRole: policy.budgetShareByRole,
    phaseBias: policy.phaseBias,
    preferYoungPlayers: policy.preferYoungPlayers,
    maxAgePreference: policy.maxAgePreference,
    preferHighStartProbability: policy.preferHighStartProbability,
    minStartProbability: policy.minStartProbability,
    preferHighVariance: policy.preferHighVariance,
    preferMultiRole: policy.preferMultiRole,
    minNumRoles: policy.minNumRoles,
    budgetShareByBlock: policy.budgetShareByBlock,
    maxTopTierCount: policy.maxTopTierCount,
    targetTopTierCount: policy.targetTopTierCount,
    avoidTopTierEarly: policy.avoidTopTierEarly,
    adaptive: policy.adaptive,
    adaptOn: policy.adaptOn,
  };
}

interface ParticipantRow {
  id: string;
  name: string;
  stats: ParticipantSimStats;
}

@Component({
  selector: 'app-auction-simulation',
  standalone: true,
  imports: [FormsModule, DecimalPipe, PercentPipe],
  template: `
    <section class="sim-panel">
      <header class="sim-header">
        <div>
          <p class="card-section-label">Simulazioni Monte Carlo</p>
          <p class="sim-subtitle">
            Stima spesa, valore rosa e probabilità di completamento con bot guidati dai preset.
            Stateless: non avvia una sessione live.
          </p>
        </div>
      </header>

      <div class="sim-controls">
        <div class="field-group">
          <label class="field-label" for="sim-n">Numero simulazioni</label>
          <div class="slider-row">
            <input id="sim-n" type="range" min="50" max="500" step="10"
              [ngModel]="nSimulations()" (ngModelChange)="nSimulations.set(+$event)" />
            <span class="slider-value">{{ nSimulations() }}</span>
          </div>
        </div>

        <div class="field-group">
          <label class="field-label">Preset bot avversari</label>
          <div class="bot-grid">
            @for (bot of botSlots(); track bot.participantId; let i = $index) {
              <div class="bot-row">
                <span class="bot-name">{{ bot.displayName }}</span>
                <select class="field-input" [ngModel]="botPresetIds()[i]" (ngModelChange)="setBotPreset(i, $event)">
                  @for (p of availablePresets; track p.id) {
                    <option [ngValue]="p.id">{{ p.labelIt }}</option>
                  }
                </select>
              </div>
            }
          </div>
          <p class="hint">Il primo partecipante (tu) usa un profilo bilanciato; gli altri N−1 sono bot.</p>
        </div>

        <div class="field-group">
          <label class="field-label" for="sim-targets">Player obiettivo (id, separati da virgola)</label>
          <input id="sim-targets" class="field-input" type="text" placeholder="es. a1, c2"
            [ngModel]="targetIdsRaw()" (ngModelChange)="targetIdsRaw.set($event)" />
        </div>

        <button type="button" class="run-btn" [disabled]="loading() || !canRun()" (click)="run()">
          {{ loading() ? 'Simulazione in corso…' : 'Simula' }}
        </button>
        @if (error(); as err) {
          <p class="error-text" role="alert">{{ err }}</p>
        }
      </div>

      @if (result(); as res) {
        <div class="sim-results">
          <p class="sim-meta">
            Completate <strong>{{ res.nCompleted }}</strong> scenari in
            {{ res.wallTimeSeconds | number: '1.1-1' }}s
            @if (res.warnings.length) { · {{ res.warnings.length }} warning }
            @if ((res.nExcludedNoProjection ?? 0) > 0) {
              · {{ res.nExcludedNoProjection }} esclusi (no proiezione)
            }
          </p>
          <p class="hint click-hint">Clicca una card per la rosa tipo (roster rappresentativo per ruolo).</p>

          <div class="stats-grid">
            @for (row of participantRows(); track row.id) {
              <button
                type="button"
                class="stat-card"
                [class.stat-card--active]="selectedId() === row.id"
                (click)="toggleDetail(row.id)"
                [attr.aria-expanded]="selectedId() === row.id"
                [attr.aria-label]="'Dettaglio simulazione ' + row.name"
              >
                <div class="stat-card__head">
                  <h4 class="stat-title">{{ row.name }}</h4>
                  <span class="stat-badge" [class.ok]="row.stats.completionProbability >= 0.9">
                    {{ row.stats.completionProbability | percent: '1.0-0' }} rosa
                  </span>
                </div>

                <div class="metric">
                  <div class="metric__label">
                    Completamento rosa
                    <span class="info" [title]="help.completion">?</span>
                  </div>
                  <p class="metric__desc">{{ help.completion }}</p>
                  <div class="completion-track">
                    <div class="completion-fill" [style.width.%]="row.stats.completionProbability * 100"></div>
                  </div>
                </div>

                <div class="metric">
                  <div class="metric__label">
                    Spesa crediti
                    <span class="info" [title]="help.spend">?</span>
                  </div>
                  <p class="metric__desc">{{ help.spend }}</p>
                  <div class="pct-bar">
                    <div class="pct-fill" [style.width.%]="spendWidth(row.stats.spendP50)"></div>
                    <div class="pct-marker" [style.left.%]="spendWidth(row.stats.spendP10)"></div>
                    <div class="pct-marker pct-marker--hi" [style.left.%]="spendWidth(row.stats.spendP90)"></div>
                  </div>
                  <span class="bar-values">
                    p10 {{ row.stats.spendP10 | number: '1.0-0' }}
                    · p50 {{ row.stats.spendP50 | number: '1.0-0' }}
                    · p90 {{ row.stats.spendP90 | number: '1.0-0' }} cr
                  </span>
                </div>

                <div class="metric">
                  <div class="metric__label">
                    Valore rosa (ESV)
                    <span class="info" [title]="help.esv">?</span>
                  </div>
                  <p class="metric__desc">{{ help.esv }}</p>
                  <div class="pct-bar pct-bar--esv">
                    <div class="pct-fill" [style.width.%]="esvWidth(row.stats.esvTotalP50)"></div>
                    <div class="pct-marker" [style.left.%]="esvWidth(row.stats.esvTotalP10)"></div>
                    <div class="pct-marker pct-marker--hi" [style.left.%]="esvWidth(row.stats.esvTotalP90)"></div>
                  </div>
                  <span class="bar-values">
                    p10 {{ row.stats.esvTotalP10 | number: '1.1-1' }}
                    · p50 {{ row.stats.esvTotalP50 | number: '1.1-1' }}
                    · p90 {{ row.stats.esvTotalP90 | number: '1.1-1' }}
                  </span>
                </div>
              </button>
            }
          </div>

          @if (selectedRow(); as sel) {
            <aside class="detail-panel" role="region" [attr.aria-label]="'Dettaglio ' + sel.name">
              <div class="detail-panel__head">
                <div>
                  <h3 class="detail-title">{{ sel.name }}</h3>
                  <p class="hint">
                    Rosa tipo bilanciata per ruolo (quote P/D/C/A) sulle {{ res.nCompleted }} simulazioni.
                  </p>
                </div>
                <button type="button" class="secondary-btn" (click)="selectedId.set(null)">Chiudi</button>
              </div>

              <div class="detail-grid">
                <div class="detail-block">
                  <p class="card-section-label">Composizione rosa (media)</p>
                  <p class="hint">Numero medio di giocatori per ruolo a fine asta.</p>
                  <div class="role-chips">
                    @for (entry of roleEntries(sel.stats); track entry.role) {
                      <span class="role-chip"><strong>{{ entry.role }}</strong> × {{ entry.count }}</span>
                    } @empty {
                      <span class="hint">Nessun dato di composizione.</span>
                    }
                  </div>
                </div>

                <div class="detail-block detail-block--wide">
                  <p class="card-section-label">Rosa tipo</p>
                  <p class="hint">
                    Roster rappresentativo: per ogni ruolo prendiamo i giocatori che questo manager
                    ha comprato più spesso, fino a riempire la quota (P/D/C/A). Non è una singola
                    asta, ma il profilo più stabile sulle N simulazioni.
                  </p>
                  @if (sel.stats.typicalSquad?.length) {
                    <div class="player-table-wrap">
                      <table class="player-table rt-card-table">
                        <thead>
                          <tr>
                            <th>Giocatore</th>
                            <th>Ruolo</th>
                            <th>Frequenza</th>
                            <th>Prezzo medio</th>
                          </tr>
                        </thead>
                        <tbody>
                          @for (pl of sel.stats.typicalSquad; track pl.playerId) {
                            <tr>
                              <td data-label="Giocatore">
                                <span class="pl-name">{{ pl.name }}</span>
                                <code class="pl-id">{{ pl.playerId }}</code>
                              </td>
                              <td data-label="Ruolo">{{ pl.role }}</td>
                              <td data-label="Frequenza">{{ pl.frequency | percent: '1.0-1' }}</td>
                              <td data-label="Prezzo medio">{{ pl.avgPrice | number: '1.0-1' }} cr</td>
                            </tr>
                          }
                        </tbody>
                      </table>
                    </div>
                  } @else {
                    <p class="hint">Nessun acquisto registrato per questo manager.</p>
                  }
                </div>
              </div>
            </aside>
          }

          @if (targetAcquisitions().length) {
            <div class="acq-block">
              <p class="card-section-label">Probabilità acquisizione (obiettivi)</p>
              <p class="hint">Su tutti i manager: quanto spesso il player finisce assegnato in un’asta simulata.</p>
              <table class="player-table rt-card-table">
                <thead>
                  <tr><th>Player</th><th>Prob.</th><th>Prezzo medio</th></tr>
                </thead>
                <tbody>
                  @for (row of targetAcquisitions(); track row.playerId) {
                    <tr>
                      <td data-label="Player"><code>{{ row.playerId }}</code></td>
                      <td data-label="Prob.">{{ row.prob | percent: '1.0-1' }}</td>
                      <td data-label="Prezzo medio">{{ row.avgPrice | number: '1.0-1' }} cr</td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          }
        </div>
      }
    </section>
  `,
  styleUrls: ['./auction-simulation.component.scss'],
})
export class AuctionSimulationComponent {
  private readonly auctionService = inject(AuctionService);

  readonly participants = input.required<AuctionParticipantSetup[]>();
  readonly config = input.required<AuctionConfig>();
  readonly seasonStart = input.required<number>();

  readonly availablePresets: readonly AuctionPreset[] = AUCTION_PRESETS;
  readonly help = METRIC_HELP;

  readonly nSimulations = signal(200);
  readonly botPresetIds = signal<string[]>([]);
  readonly targetIdsRaw = signal('');
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly result = signal<AuctionSimulationResponse | null>(null);
  readonly selectedId = signal<string | null>(null);

  readonly botSlots = computed(() => this.participants().slice(1));

  readonly canRun = computed(() => {
    const parts = this.participants();
    const cfg = this.config();
    return parts.length >= 1 && !!cfg?.numParticipants && cfg.numParticipants >= 1;
  });

  readonly participantRows = computed((): ParticipantRow[] => {
    const res = this.result();
    if (!res) return [];
    const parts = this.participants();
    return Object.entries(res.perParticipant).map(([id, stats]) => ({
      id,
      name: parts.find((p) => p.participantId === id)?.displayName ?? id,
      stats,
    }));
  });

  readonly selectedRow = computed((): ParticipantRow | null => {
    const id = this.selectedId();
    if (!id) return null;
    return this.participantRows().find((r) => r.id === id) ?? null;
  });

  readonly targetAcquisitions = computed(() => {
    const res = this.result();
    if (!res) return [];
    const ids = this.targetIdsRaw()
      .split(/[,;\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (!ids.length) return [];
    return ids
      .map((playerId) => {
        const stats = res.playerAcquisitionProbability[playerId];
        return stats
          ? { playerId, prob: stats.prob, avgPrice: stats.avgPrice }
          : { playerId, prob: 0, avgPrice: 0 };
      })
      .sort((a, b) => b.prob - a.prob);
  });

  private maxSpend = 1;
  private maxEsv = 1;

  constructor() {
    effect(() => {
      this.participants();
      this.ensureBotPresets();
    });
  }

  ensureBotPresets(): void {
    const bots = this.botSlots();
    const current = this.botPresetIds();
    if (current.length === bots.length) return;
    this.botPresetIds.set(
      bots.map((_, i) => current[i] ?? DEFAULT_BOT_PRESET_IDS[i % DEFAULT_BOT_PRESET_IDS.length]),
    );
  }

  setBotPreset(index: number, presetId: string): void {
    this.ensureBotPresets();
    const next = [...this.botPresetIds()];
    next[index] = presetId;
    this.botPresetIds.set(next);
  }

  toggleDetail(id: string): void {
    this.selectedId.update((cur) => (cur === id ? null : id));
  }

  roleEntries(stats: ParticipantSimStats): { role: string; count: number }[] {
    return Object.entries(stats.squadCompositionMode ?? {})
      .map(([role, count]) => ({ role, count }))
      .sort((a, b) => a.role.localeCompare(b.role));
  }

  spendWidth(value: number): number {
    return Math.max(0, Math.min(100, (value / Math.max(this.maxSpend, 1)) * 100));
  }

  esvWidth(value: number): number {
    return Math.max(0, Math.min(100, (Math.max(0, value) / Math.max(this.maxEsv, 1)) * 100));
  }

  run(): void {
    this.ensureBotPresets();
    this.error.set(null);
    this.loading.set(true);
    this.result.set(null);
    this.selectedId.set(null);

    const parts = this.participants();
    const cfg = this.config();
    const botIds = this.botPresetIds();

    const bidderProfiles: BidderProfile[] = parts.map((p, i) => {
      if (i === 0) {
        const balanced = findAuctionPreset('balanced');
        return { participantId: p.participantId, policy: policyFromPreset(balanced?.policy) };
      }
      const preset = findAuctionPreset(botIds[i - 1] ?? 'balanced');
      return { participantId: p.participantId, policy: policyFromPreset(preset?.policy) };
    });

    const req: SimulateAuctionRequest = {
      seasonStart: this.seasonStart(),
      participants: parts,
      config: cfg,
      bidderProfiles,
      simConfig: { nSimulations: this.nSimulations(), randomSeed: 42 },
    };

    this.auctionService.simulate(req).subscribe({
      next: (res) => {
        let ms = 1;
        let me = 1;
        for (const s of Object.values(res.perParticipant)) {
          ms = Math.max(ms, s.spendP90, s.spendP50);
          me = Math.max(me, s.esvTotalP90, s.esvTotalP50, 1);
        }
        this.maxSpend = ms;
        this.maxEsv = me;
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Simulazione fallita';
        this.error.set(typeof detail === 'string' ? detail : JSON.stringify(detail));
        this.loading.set(false);
      },
    });
  }
}
