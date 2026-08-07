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

@Component({
  selector: 'app-auction-simulation',
  standalone: true,
  imports: [FormsModule, DecimalPipe, PercentPipe],
  template: `
    <section class="sim-panel card">
      <header class="sim-header">
        <div>
          <p class="card-section-label">Simulazioni Monte Carlo</p>
          <p class="sim-subtitle muted">
            Stima spesa, ESV e probabilità di completamento rosa con bot avversari
            guidati dai preset strategici. Stateless — non avvia una sessione live.
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
          <p class="muted small">Il primo partecipante (tu) usa un profilo neutro; gli altri N−1 sono bot.</p>
        </div>

        <div class="field-group">
          <label class="field-label" for="sim-targets">Player obiettivo (id, separati da virgola)</label>
          <input id="sim-targets" class="field-input" type="text" placeholder="es. a1, c2"
            [ngModel]="targetIdsRaw()" (ngModelChange)="targetIdsRaw.set($event)" />
        </div>

        <button type="button" class="primary-btn full-w" [disabled]="loading() || !canRun()" (click)="run()">
          {{ loading() ? 'Simulazione in corso…' : 'Simula' }}
        </button>
        @if (error(); as err) { <p class="error-text" role="alert">{{ err }}</p> }
      </div>

      @if (result(); as res) {
        <div class="sim-results">
          <p class="muted small">
            Completate {{ res.nCompleted }} scenari in {{ res.wallTimeSeconds | number: '1.2-2' }}s
            @if (res.warnings.length) { · {{ res.warnings.length }} warning }
          </p>
          <div class="stats-grid">
            @for (row of participantRows(); track row.id) {
              <article class="stat-card">
                <h4 class="stat-title">{{ row.name }}</h4>
                <p class="stat-line">Completamento rosa: <strong>{{ row.stats.completionProbability | percent: '1.0-0' }}</strong></p>
                <div class="bar-block">
                  <span class="bar-label">Spesa (p10 / p50 / p90)</span>
                  <div class="pct-bar">
                    <div class="pct-fill p50" [style.width.%]="spendWidth(row.stats.spendP50)"></div>
                    <div class="pct-marker p10" [style.left.%]="spendWidth(row.stats.spendP10)"></div>
                    <div class="pct-marker p90" [style.left.%]="spendWidth(row.stats.spendP90)"></div>
                  </div>
                  <span class="bar-values">
                    {{ row.stats.spendP10 | number: '1.0-0' }} / {{ row.stats.spendP50 | number: '1.0-0' }} / {{ row.stats.spendP90 | number: '1.0-0' }} cr
                  </span>
                </div>
                <div class="bar-block">
                  <span class="bar-label">ESV totale (p10 / p50 / p90)</span>
                  <div class="pct-bar esv">
                    <div class="pct-fill p50" [style.width.%]="esvWidth(row.stats.esvTotalP50)"></div>
                    <div class="pct-marker p10" [style.left.%]="esvWidth(row.stats.esvTotalP10)"></div>
                    <div class="pct-marker p90" [style.left.%]="esvWidth(row.stats.esvTotalP90)"></div>
                  </div>
                  <span class="bar-values">
                    {{ row.stats.esvTotalP10 | number: '1.1-1' }} / {{ row.stats.esvTotalP50 | number: '1.1-1' }} / {{ row.stats.esvTotalP90 | number: '1.1-1' }}
                  </span>
                </div>
              </article>
            }
          </div>
          @if (targetAcquisitions().length) {
            <div class="acq-table-wrap">
              <p class="card-section-label">Probabilità acquisizione (obiettivi)</p>
              <table class="acq-table">
                <thead><tr><th>Player</th><th>Prob.</th><th>Prezzo medio</th></tr></thead>
                <tbody>
                  @for (row of targetAcquisitions(); track row.playerId) {
                    <tr>
                      <td><code>{{ row.playerId }}</code></td>
                      <td>{{ row.prob | percent: '1.0-1' }}</td>
                      <td>{{ row.avgPrice | number: '1.0-1' }} cr</td>
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
  readonly nSimulations = signal(200);
  readonly botPresetIds = signal<string[]>([]);
  readonly targetIdsRaw = signal('');
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly result = signal<AuctionSimulationResponse | null>(null);
  readonly botSlots = computed(() => this.participants().slice(1));
  readonly canRun = computed(() => {
    const parts = this.participants();
    const cfg = this.config();
    return parts.length >= 1 && !!cfg?.numParticipants && cfg.numParticipants >= 1;
  });
  readonly participantRows = computed(() => {
    const res = this.result();
    if (!res) return [] as { id: string; name: string; stats: ParticipantSimStats }[];
    const parts = this.participants();
    return Object.entries(res.perParticipant).map(([id, stats]) => ({
      id, name: parts.find((p) => p.participantId === id)?.displayName ?? id, stats,
    }));
  });
  readonly targetAcquisitions = computed(() => {
    const res = this.result();
    if (!res) return [];
    const ids = this.targetIdsRaw().split(/[,;\s]+/).map((s) => s.trim()).filter(Boolean);
    if (!ids.length) return [];
    return ids.map((playerId) => {
      const stats = res.playerAcquisitionProbability[playerId];
      return stats ? { playerId, prob: stats.prob, avgPrice: stats.avgPrice } : { playerId, prob: 0, avgPrice: 0 };
    }).sort((a, b) => b.prob - a.prob);
  });
  private maxSpend = 1;
  private maxEsv = 1;

  constructor() {
    effect(() => { this.participants(); this.ensureBotPresets(); });
  }

  ensureBotPresets(): void {
    const bots = this.botSlots();
    const current = this.botPresetIds();
    if (current.length === bots.length) return;
    this.botPresetIds.set(bots.map((_, i) => current[i] ?? DEFAULT_BOT_PRESET_IDS[i % DEFAULT_BOT_PRESET_IDS.length]));
  }

  setBotPreset(index: number, presetId: string): void {
    this.ensureBotPresets();
    const next = [...this.botPresetIds()];
    next[index] = presetId;
    this.botPresetIds.set(next);
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
        let ms = 1, me = 1;
        for (const s of Object.values(res.perParticipant)) {
          ms = Math.max(ms, s.spendP90, s.spendP50);
          me = Math.max(me, s.esvTotalP90, s.esvTotalP50, 1);
        }
        this.maxSpend = ms; this.maxEsv = me;
        this.result.set(res); this.loading.set(false);
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Simulazione fallita';
        this.error.set(typeof detail === 'string' ? detail : JSON.stringify(detail));
        this.loading.set(false);
      },
    });
  }
}
