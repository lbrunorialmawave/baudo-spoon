import { Component, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { OverviewPlayer, SortKey } from '../../../../core/models/overview.models';
import { FASE7_LABELS, FASE7_TOOLTIPS, HYBRID_LABELS } from '../../../../core/models/mantra.models';
import { TitolaritaBadgesComponent } from '../titolarita-badges/titolarita-badges.component';
import { SkeletonComponent } from '../../../../shared/components/skeleton/skeleton.component';

/** Column set mirrors player-table.component.ts's responsive pattern
 *  (progressive `hidden sm/md/lg:table-cell` disclosure inside a
 *  horizontally-scrollable container) — same vocabulary, not a new one.
 *  Always visible: #, Player, FP Ibrido, Titolarità, Profilo — the rest
 *  reveals as the viewport grows or via horizontal scroll. */
@Component({
  selector: 'app-overview-table',
  standalone: true,
  imports: [SkeletonComponent, DecimalPipe, TitolaritaBadgesComponent],
  template: `
    <div class="overflow-x-auto -mx-3 sm:-mx-4 md:mx-0">
      <table class="w-full text-sm" style="border-collapse:collapse;min-width:760px">
        <thead>
          <tr class="border-b text-xs font-medium uppercase tracking-wide"
              style="border-color:var(--color-border);color:var(--color-text-secondary)">
            <th class="px-3 py-2 text-right w-10 sm:w-12">#</th>
            <th class="px-3 py-2 text-left sortable" (click)="sortChanged.emit({ column: 'player_name', additive: $event.shiftKey })" [title]="sortHint()">
              Player @if (sortDirFor('player_name'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('player_name') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left sortable hidden sm:table-cell" (click)="sortChanged.emit({ column: 'team', additive: $event.shiftKey })" [title]="sortHint()">
              Team @if (sortDirFor('team'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('team') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left sortable hidden md:table-cell" (click)="sortChanged.emit({ column: 'ruolo_primario', additive: $event.shiftKey })" [title]="'Ruolo primario / ruoli secondari nel sistema Mantra · ' + sortHint()">
              Ruolo @if (sortDirFor('ruolo_primario'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('ruolo_primario') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable" (click)="sortChanged.emit({ column: 'fpIbrido', additive: $event.shiftKey })" [title]="'FP Ibrido — punteggio combinato MANTRA + Machine Learning · ' + sortHint()">
              FP Ibrido @if (sortDirFor('fpIbrido'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('fpIbrido') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit({ column: 'FP_Mantra', additive: $event.shiftKey })" [title]="'FP calcolato solo dal sistema MANTRA (voti storici) · ' + sortHint()">
              FP Mantra @if (sortDirFor('FP_Mantra'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('FP_Mantra') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit({ column: 'predicted_fantavoto', additive: $event.shiftKey })" [title]="'Fantavoto previsto dal modello Machine Learning · ' + sortHint()">
              Voto ML @if (sortDirFor('predicted_fantavoto'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('predicted_fantavoto') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit({ column: 'confidenceScore', additive: $event.shiftKey })" [title]="'Affidabilità della prediction ML (0-100) · ' + sortHint()">
              Conf. @if (sortDirFor('confidenceScore'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('confidenceScore') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable hidden lg:table-cell" (click)="sortChanged.emit({ column: 'VR', additive: $event.shiftKey })" [title]="'Valore Reale — indice di convenienza prezzo/valore · ' + sortHint()">
              VR @if (sortDirFor('VR'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('VR') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left" title="Titolarità: Reale (probabili formazioni) · ML · Gruppo Esperti — 3 segnali distinti, non ordinabile come colonna unica">Titolarità</th>
            <th class="px-3 py-2 text-left sortable" (click)="sortChanged.emit({ column: 'Fase7', additive: $event.shiftKey })" [title]="sortHint()">
              Profilo @if (sortDirFor('Fase7'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('Fase7') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left hidden lg:table-cell" title="Segnali di classificazione ibrida MANTRA vs ML — non ordinabile (più etichette per riga)">Segnali</th>
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit({ column: 'Pz1', additive: $event.shiftKey })" [title]="'Quotazione ufficiale corrente del listone · ' + sortHint()">
              Prezzo @if (sortDirFor('Pz1'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('Pz1') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left sortable hidden lg:table-cell" (click)="sortChanged.emit({ column: 'expert_totale', additive: $event.shiftKey })" [title]="sortHint()">
              Esperti @if (sortDirFor('expert_totale'); as dir) { <span style="color:var(--color-accent)"><sup>{{ sortRankLabel('expert_totale') }}</sup>{{ dir === 'asc' ? '▲' : '▼' }}</span> }
            </th>
          </tr>
        </thead>
        <tbody>
          @if (loading()) {
            @for (_ of skeletonRows; track $index) {
              <tr>
                @for (__ of [1,2,3,4,5,6,7,8,9,10,11,12,13]; track $index) {
                  <td class="px-3 py-2"><app-skeleton height="20px" /></td>
                }
              </tr>
            }
          } @else {
            @for (item of items(); track item.fantacalcioId; let i = $index) {
              <tr class="border-b cursor-pointer transition-colors"
                  style="border-color:var(--color-border)"
                  (click)="playerSelected.emit(item)"
                  (mouseenter)="hoverId = item.fantacalcioId"
                  (mouseleave)="hoverId = null"
                  [style.backgroundColor]="hoverId === item.fantacalcioId ? 'var(--color-surface)' : 'transparent'">
                <td class="px-3 py-2.5 text-right font-mono text-xs" style="color:var(--color-text-secondary)">
                  {{ (page() - 1) * pageSize() + i + 1 }}
                </td>
                <td class="px-3 py-2.5 font-medium" style="color:var(--color-text-primary)">
                  {{ item.playerName ?? '—' }}
                </td>
                <td class="px-3 py-2.5 text-xs hidden sm:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.team ?? '—' }}
                </td>
                <td class="px-3 py-2.5 text-xs hidden md:table-cell whitespace-nowrap">
                  @if (roles(item); as r) {
                    <span class="font-medium" style="color:var(--color-text-primary)">{{ r.primary }}</span>
                    @if (r.secondary.length) {
                      <span style="color:var(--color-text-secondary)">/{{ r.secondary.join('/') }}</span>
                    }
                  } @else {
                    <span class="italic opacity-50">—</span>
                  }
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-sm font-semibold" style="color:var(--color-accent)">
                  {{ item.fpIbrido != null ? (item.fpIbrido | number:'1.1-1') : '—' }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden md:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.FP_Mantra != null ? (item.FP_Mantra | number:'1.1-1') : '—' }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden md:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.predictedFantavoto != null ? (item.predictedFantavoto | number:'1.2-2') : '—' }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden md:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.confidenceScore != null ? (item.confidenceScore | number:'1.0-0') : '—' }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.VR != null ? (item.VR | number:'1.0-0') : '—' }}
                </td>
                <td class="px-3 py-2.5">
                  <app-titolarita-badges size="sm"
                    [statusScraped]="item.statusScraped"
                    [probabilityScraped]="item.probabilityScraped"
                    [startProbability]="item.startProbability"
                    [expertTitolarita]="item.expertTitolarita" />
                </td>
                <td class="px-3 py-2.5">
                  @if (item.Fase7) {
                    @let f7 = FASE7_LABELS[item.Fase7];
                    <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                          [style.background]="f7?.color ?? '#6B7280'"
                          [title]="FASE7_TOOLTIPS[item.Fase7]">
                      <span class="sm:hidden">{{ f7?.icon ?? '' }}</span>
                      <span class="hidden sm:inline">{{ f7?.icon ?? '' }} {{ f7?.label ?? item.Fase7 }}</span>
                    </span>
                  } @else {
                    <span class="rounded-full border px-2 py-0.5 text-xs font-medium"
                          style="border-color:var(--color-border);color:var(--color-text-secondary)">
                      ➖
                    </span>
                  }
                </td>
                <td class="px-3 py-2.5 hidden lg:table-cell whitespace-nowrap">
                  @if (item.hybridLabels?.length) {
                    <span class="inline-flex items-center gap-1">
                      @for (id of item.hybridLabels!.slice(0, 2); track id) {
                        @let meta = labelMeta(id);
                        <span class="rounded-full px-1.5 py-0.5 text-[10px] font-medium text-white"
                              [style.background]="meta?.color ?? '#6B7280'"
                              [title]="meta?.desc ?? id">
                          {{ meta?.label ?? id }}
                        </span>
                      }
                      @if (item.hybridLabels!.length > 2) {
                        <span class="text-xs opacity-50">+{{ item.hybridLabels!.length - 2 }}</span>
                      }
                    </span>
                  } @else {
                    <span class="text-xs opacity-30">—</span>
                  }
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden md:table-cell whitespace-nowrap" style="color:var(--color-text-secondary)">
                  {{ item.Pz1 != null ? (item.Pz1 | number:'1.0-0') + ' cr' : '—' }}
                </td>
                <td class="px-3 py-2.5 text-xs hidden lg:table-cell whitespace-nowrap">
                  @if (item.expertRating != null) {
                    <span style="color:var(--color-accent)" [title]="expertTooltip(item)">
                      {{ stars(item.expertRating) }}
                      @if (item.expertTotale != null) {
                        <span class="opacity-60">{{ item.expertTotale }}/50</span>
                      }
                    </span>
                  } @else {
                    <span class="opacity-30">—</span>
                  }
                </td>
              </tr>
            }
          }
        </tbody>
      </table>
    </div>
  `,
  styles: [`
    :host { display: block; }
    .sortable { cursor: pointer; user-select: none; }
    .sortable:hover { color: var(--color-accent) !important; }
  `],
})
export class OverviewTableComponent {
  readonly items = input.required<OverviewPlayer[]>();
  readonly loading = input<boolean>(false);
  readonly page = input<number>(1);
  readonly pageSize = input<number>(50);
  readonly sortKeys = input<SortKey[]>([]);
  readonly sortChanged = output<{ column: string; additive: boolean }>();
  readonly playerSelected = output<OverviewPlayer>();

  hoverId: number | null = null;
  readonly skeletonRows = Array.from({ length: 8 });

  readonly FASE7_LABELS = FASE7_LABELS;
  readonly FASE7_TOOLTIPS = FASE7_TOOLTIPS;

  private static readonly SUPERSCRIPTS = ['¹', '²', '³'];

  readonly sortHint = (): string =>
    this.sortKeys().length
      ? 'Click per ordinare, Shift+click per combinare più criteri'
      : 'Click per ordinare, Shift+click per aggiungere un secondo criterio';

  /** Direction for a column if it's an active sort key, else null. */
  readonly sortDirFor = (column: string): 'asc' | 'desc' | null =>
    this.sortKeys().find(k => k.column === column)?.direction ?? null;

  /** Priority superscript (¹²³) for a column, only once ≥2 criteria are
   *  active — a single active key needs no rank, the arrow alone reads fine. */
  readonly sortRankLabel = (column: string): string => {
    if (this.sortKeys().length < 2) return '';
    const idx = this.sortKeys().findIndex(k => k.column === column);
    return idx === -1 ? '' : (OverviewTableComponent.SUPERSCRIPTS[idx] ?? '');
  };

  readonly stars = (rating: number): string =>
    '★'.repeat(rating) + '☆'.repeat(Math.max(0, 5 - rating));

  /** Primary role first, followed by any other Mantra roles the player
   *  covers — same column, "/"-joined per user request (was a separate
   *  "Category" column in Players). */
  readonly roles = (item: OverviewPlayer): { primary: string; secondary: string[] } | null => {
    if (!item.ruoloPrimario) return null;
    const secondary = (item.ruoliMantra ?? []).filter(r => r !== item.ruoloPrimario);
    return { primary: item.ruoloPrimario, secondary };
  };

  readonly labelMeta = (id: string) => HYBRID_LABELS.find(l => l.id === id);

  readonly expertTooltip = (p: OverviewPlayer): string => {
    const parts: string[] = [];
    if (p.expertTotale != null) {
      parts.push(
        `Titolarità ${p.expertTitolarita}/10 · Media voto ${p.expertMediaVoto}/10 · ` +
        `Salute ${p.expertSalute}/10 · ${p.expertBonusLabel ?? 'Bonus'} ${p.expertBonusValue}/10 · ` +
        `TOTALE ${p.expertTotale}/50`
      );
    }
    if (p.expertComment) parts.push(p.expertComment);
    return parts.join('\n');
  };
}
