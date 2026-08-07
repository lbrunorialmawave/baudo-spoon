import { Component, effect, inject, input, output, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { OverviewPlayer } from '../../../../core/models/overview.models';
import { FASE7_LABELS } from '../../../../core/models/mantra.models';
import { PlayerSeasonStat } from '../../../../core/models/stats.models';
import { PlayerQuotation } from '../../../../core/models/quotations.models';
import { NextSeasonPrediction } from '../../../../core/models/api.models';
import { StatsService } from '../../../../core/services/stats.service';
import { QuotationService } from '../../../../core/services/quotation.service';
import { PredictionService } from '../../../../core/services/prediction.service';
import { DrawerShellComponent } from '../../../../shared/components/drawer-shell/drawer-shell.component';
import { TitolaritaBadgesComponent } from '../titolarita-badges/titolarita-badges.component';
import { SkeletonComponent } from '../../../../shared/components/skeleton/skeleton.component';

/** Full player detail for the Overview page: MANTRA + ML + Ibrido (all
 *  already on the row, no fetch needed — same as PredictionDrawerComponent)
 *  plus Titolarità comparison, Gruppo Esperti (row summary + fetched
 *  history), and the historic sections (stats/quotation/next-season),
 *  fetched keyed on `playerFotmobId` — which the /overview/players
 *  endpoint always populates from the source MANTRA data, unlike the
 *  Players page's drawer today. */
@Component({
  selector: 'app-overview-drawer',
  standalone: true,
  imports: [DecimalPipe, DrawerShellComponent, TitolaritaBadgesComponent, SkeletonComponent],
  template: `
    <app-drawer-shell
      [title]="player().playerName ?? '—'"
      [subtitle]="subtitle()"
      (closed)="closed.emit()">

      <a [href]="fantacalcioUrl()" target="_blank" rel="noopener"
         class="inline-flex items-center gap-1 text-xs mb-4 hover:underline" style="color:var(--color-accent)">
        Vai a Fantacalcio.it ↗
      </a>

      @if (player().Fase7 || (player().hybridLabels?.length ?? 0) > 0) {
        <section class="mb-5">
          <div class="flex flex-wrap gap-1.5">
            @if (player().Fase7; as f7) {
              @let f7meta = FASE7_LABELS[f7];
              <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                    [style.background]="f7meta?.color ?? '#6B7280'">
                {{ f7meta?.icon ?? '' }} {{ f7meta?.label ?? f7 }}
              </span>
            }
            @for (l of player().hybridLabels ?? []; track l) {
              <span class="rounded-full px-2 py-0.5 text-xs font-medium"
                    style="background:var(--color-surface-raised);color:var(--color-text-secondary)">
                {{ l }}
              </span>
            }
          </div>
        </section>
      }

      <section class="mb-5">
        <h3 class="section-title">Titolarità</h3>
        <app-titolarita-badges size="lg"
          [statusScraped]="player().statusScraped"
          [probabilityScraped]="player().probabilityScraped"
          [startProbability]="player().startProbability"
          [expertTitolarita]="player().expertTitolarita" />
      </section>

      <section class="mb-5">
        <h3 class="section-title">Punteggi MANTRA</h3>
        <div class="grid grid-cols-2 gap-2">
          @for (row of mantraRows(); track row.label) {
            <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
              <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
              <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                {{ row.value != null ? (row.value | number:'1.1-1') : '—' }}
              </p>
            </div>
          }
        </div>
      </section>

      <section class="mb-5">
        <h3 class="section-title">Machine Learning</h3>
        <div class="grid grid-cols-2 gap-2">
          @for (row of mlRows(); track row.label) {
            <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
              <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
              <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                {{ row.value != null ? (row.value | number:'1.1-1') : '—' }}
              </p>
            </div>
          }
        </div>
        @if (!player().hasMlData) {
          <p class="text-xs mt-2" style="color:var(--color-text-secondary)">
            Nessun dato ML disponibile per questo giocatore.
          </p>
        }
      </section>

      <section class="mb-5">
        <h3 class="section-title">Ibrido</h3>
        <div class="grid grid-cols-2 gap-2">
          @for (row of hybridRows(); track row.label) {
            <div class="rounded-lg px-3 py-2" style="background:var(--color-surface)">
              <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">{{ row.label }}</p>
              <p class="font-mono text-sm font-semibold" style="color:var(--color-text-primary)">
                {{ row.value != null ? (row.value | number:'1.1-1') : '—' }}
              </p>
            </div>
          }
        </div>
      </section>

      <!-- Gruppo Esperti: row summary (zero-latency, already on the payload) + fetched history -->
      <section class="mb-5">
        <h3 class="section-title">Gruppo Esperti</h3>
        @if (player().expertTotale != null) {
          <div class="rounded-lg p-3 mb-2" style="background:var(--color-surface)">
            <div class="flex items-center justify-between gap-2">
              <span class="text-xs font-medium" style="color:var(--color-text-secondary)">
                {{ player().expertName ?? 'Gruppo Esperti' }}
              </span>
              @if (player().expertRating != null) {
                <span class="text-sm tracking-tight" style="color:var(--color-accent)">
                  {{ stars(player().expertRating!) }}
                </span>
              }
            </div>
            <div class="flex flex-wrap gap-x-3 gap-y-0.5 mt-1.5 text-xs" style="color:var(--color-text-secondary)">
              <span>Titolarità {{ player().expertTitolarita }}/10</span>
              <span>Media voto {{ player().expertMediaVoto }}/10</span>
              <span>Salute {{ player().expertSalute }}/10</span>
              <span>{{ player().expertBonusLabel ?? 'Bonus' }} {{ player().expertBonusValue }}/10</span>
              <span style="color:var(--color-text-primary)" class="font-medium">TOTALE {{ player().expertTotale }}/50</span>
            </div>
            @if (player().expertComment) {
              <p class="text-xs mt-1.5 leading-snug" style="color:var(--color-text-primary)">{{ player().expertComment }}</p>
            }
            @if (player().expertUrl) {
              <a [href]="player().expertUrl!" target="_blank" rel="noopener" class="text-xs mt-1.5 inline-block" style="color:var(--color-accent)">Fonte ↗</a>
            }
          </div>
        } @else {
          <p class="text-xs mb-2" style="color:var(--color-text-secondary)">Nessuna valutazione per questo giocatore</p>
        }
      </section>

      <!-- Next season prediction -->
      <section class="mb-5">
        <h3 class="section-title">Next Season Forecast</h3>
        @if (nextLoading()) {
          <app-skeleton height="48px" />
        } @else if (nextPred(); as pred) {
          <div class="rounded-lg p-3" style="background:var(--color-surface)">
            <p class="text-xs" style="color:var(--color-text-secondary)">Predicted fantavoto</p>
            <p class="text-2xl font-bold tabular-nums mt-0.5" style="color:var(--color-accent)">
              {{ pred.predictedNextFantavoto | number:'1.2-2' }}
            </p>
          </div>
        } @else {
          <p class="text-xs" style="color:var(--color-text-secondary)">Not available</p>
        }
      </section>

      <!-- Stats history -->
      <section class="mb-5">
        <h3 class="section-title">Stats History</h3>
        @if (statsLoading()) {
          <div class="space-y-1.5">
            @for (_ of [1,2,3]; track $index) { <app-skeleton height="36px" /> }
          </div>
        } @else if (statsHistory().length) {
          <ul class="space-y-1">
            @for (s of statsHistory(); track s.id) {
              <li class="flex items-center justify-between rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <span class="text-xs" style="color:var(--color-text-secondary)">{{ s.season.season_label }} · {{ s.stat_category }}</span>
                <span class="font-mono text-sm font-semibold" style="color:var(--color-accent)">{{ s.value ?? '—' }}</span>
              </li>
            }
          </ul>
        } @else {
          <p class="text-xs" style="color:var(--color-text-secondary)">No data</p>
        }
      </section>

      <!-- Quotation history -->
      <section>
        <h3 class="section-title">Auction Price History</h3>
        @if (quotLoading()) {
          <div class="space-y-1.5">
            @for (_ of [1,2]; track $index) { <app-skeleton height="36px" /> }
          </div>
        } @else if (quotHistory().length) {
          <ul class="space-y-1">
            @for (q of quotHistory(); track q.id) {
              <li class="flex items-center justify-between rounded-lg px-3 py-2" style="background:var(--color-surface)">
                <span class="text-xs" style="color:var(--color-text-secondary)">{{ q.seasonStart }}/{{ q.seasonStart + 1 }} · {{ q.team }}</span>
                <div class="flex items-center gap-3 text-xs">
                  <span style="color:var(--color-text-secondary)">qtA <strong style="color:var(--color-text-primary)">{{ q.qtA }}</strong></span>
                  @if (q.fvm !== null) {
                    <span style="color:var(--color-text-secondary)">fvm <strong style="color:var(--color-text-primary)">{{ q.fvm }}</strong></span>
                  }
                </div>
              </li>
            }
          </ul>
        } @else {
          <p class="text-xs" style="color:var(--color-text-secondary)">No quotation data</p>
        }
      </section>
    </app-drawer-shell>
  `,
  styles: [`
    :host { display: contents; }
    .section-title {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.06em; margin-bottom: 8px;
      color: var(--color-text-secondary);
    }
  `],
})
export class OverviewDrawerComponent {
  readonly player = input.required<OverviewPlayer>();
  readonly closed = output<void>();

  private readonly statsService = inject(StatsService);
  private readonly quotService = inject(QuotationService);
  private readonly predService = inject(PredictionService);

  readonly FASE7_LABELS = FASE7_LABELS;

  readonly statsHistory = signal<PlayerSeasonStat[]>([]);
  readonly quotHistory = signal<PlayerQuotation[]>([]);
  readonly nextPred = signal<NextSeasonPrediction | null>(null);

  readonly statsLoading = signal(false);
  readonly quotLoading = signal(false);
  readonly nextLoading = signal(false);

  readonly stars = (rating: number): string =>
    '★'.repeat(rating) + '☆'.repeat(Math.max(0, 5 - rating));

  readonly subtitle = (): string => {
    const p = this.player();
    const parts = [p.team, p.ruoloPrimario].filter((v): v is string => !!v);
    return parts.join(' · ');
  };

  /** fantacalcio.it resolves its player-detail page off the trailing
   *  numeric ID alone — team/slug segments are ignored server-side
   *  (verified live: a placeholder team/slug still serves the correct
   *  player page) — so fantacalcioId is all we need, no extra scraping. */
  readonly fantacalcioUrl = (): string =>
    `https://www.fantacalcio.it/serie-a/squadre/_/_/${this.player().fantacalcioId}`;

  mantraRows(): { label: string; value: number | null }[] {
    const p = this.player();
    return [
      { label: 'P1', value: p.P1 },
      { label: 'P2', value: p.P2 },
      { label: 'P3', value: p.P3 },
      { label: 'P4', value: p.P4 },
      { label: 'CP', value: p.CP },
      { label: 'FP', value: p.FP },
      { label: 'FP_Mantra', value: p.FP_Mantra },
      { label: 'VR', value: p.VR },
      { label: 'Prezzo Massimo', value: p.prezzoMassimo },
    ];
  }

  mlRows(): { label: string; value: number | null }[] {
    const p = this.player();
    return [
      { label: 'Fantavoto previsto', value: p.predictedFantavoto },
      { label: 'Deviazione std.', value: p.predictionStd },
      { label: 'Minuti attesi', value: p.expectedMinutes },
      { label: 'ML score (0-100)', value: p.mlScoreNorm },
      { label: 'ML boost', value: p.mlBoost },
      { label: 'Confidence', value: p.confidenceScore },
    ];
  }

  hybridRows(): { label: string; value: number | null }[] {
    const p = this.player();
    return [
      { label: 'FP Ibrido', value: p.fpIbrido },
      { label: 'Gap MANTRA-ML', value: p.fpGap },
      { label: 'Expected value', value: p.expectedValue },
      { label: 'VAR score', value: p.varScore },
      { label: 'ESV', value: p.esv },
      { label: 'Next season', value: p.nextSeasonPredicted },
    ];
  }

  constructor() {
    effect(() => {
      const p = this.player();
      const fid = p.playerFotmobId;

      if (fid == null) {
        this.statsHistory.set([]);
        this.quotHistory.set([]);
        this.nextPred.set(null);
        return;
      }

      this.statsLoading.set(true);
      this.statsService.getPlayerStatsById(fid).subscribe({
        next: items => { this.statsHistory.set(items); this.statsLoading.set(false); },
        error: () => { this.statsHistory.set([]); this.statsLoading.set(false); },
      });

      this.quotLoading.set(true);
      this.quotService.getPlayerHistory(fid).subscribe({
        next: res => { this.quotHistory.set(res.items); this.quotLoading.set(false); },
        error: () => { this.quotHistory.set([]); this.quotLoading.set(false); },
      });

      this.nextLoading.set(true);
      this.predService.getNextSeason(p.playerName ?? undefined).subscribe({
        next: items => { this.nextPred.set(items[0] ?? null); this.nextLoading.set(false); },
        error: () => { this.nextPred.set(null); this.nextLoading.set(false); },
      });
    });
  }
}
