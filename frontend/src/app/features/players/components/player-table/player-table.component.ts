import { Component, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { MantraPlayer, FASE7_LABELS, FASE7_TOOLTIPS, MATCHDAY_STATUS_CONFIG } from '../../../../core/models/mantra.models';
import { ExpertRatingWithFantacalcioId } from '../../../../core/models/expert-ratings.models';
import { SkeletonComponent } from '../../../../shared/components/skeleton/skeleton.component';

@Component({
  selector: 'app-player-table',
  standalone: true,
  imports: [SkeletonComponent, DecimalPipe],
  template: `
    <div class="overflow-x-auto -mx-3 sm:-mx-4 md:mx-0">
      <table class="w-full text-sm" style="border-collapse:collapse;min-width:560px">
        <thead>
          <tr class="border-b text-xs font-medium uppercase tracking-wide"
              style="border-color:var(--color-border);color:var(--color-text-secondary)">
            <th class="px-3 py-2 text-right w-10 sm:w-12">#</th>
            <th class="px-3 py-2 text-left sortable" (click)="sortChanged.emit('player_name')">
              Player @if (sortColumn() === 'player_name') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left sortable hidden sm:table-cell" (click)="sortChanged.emit('team')">
              Team @if (sortColumn() === 'team') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left sortable" (click)="sortChanged.emit('ruolo_primario')" title="Ruolo primario nel sistema Mantra (12 ruoli: Por, Dc, Dd, Ds, B, E, M, C, T, W, A, Pc)">
              Mantra @if (sortColumn() === 'ruolo_primario') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left hidden lg:table-cell" title="Categoria statistica / ruoli secondari Mantra">Category</th>
            <th class="px-3 py-2 text-right sortable" (click)="sortChanged.emit('FP_Mantra')" title="Fantacalcio Punti — punteggio complessivo calcolato su voti, bonus/malus e ruolo Mantra">
              FP @if (sortColumn() === 'FP_Mantra') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit('VR')" title="Valore Reale — indice di convenienza prezzo/valore (0-300, ~100 = valore equo, oltre 130 = sottovalutato dal mercato)">
              VR @if (sortColumn() === 'VR') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit('start_probability')" title="Start Probability — probabilità di essere titolare">
              SP% @if (sortColumn() === 'start_probability') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left hidden md:table-cell" title="Stato per la prossima giornata (infortunato, squalificato, etc.)">Status</th>
            <th class="px-3 py-2 text-left">
              Profilo <span class="opacity-60 cursor-help" [title]="PROFILO_LEGEND">ⓘ</span>
            </th>
            <th class="px-3 py-2 text-right sortable" (click)="sortChanged.emit('Prezzo_Massimo')" title="Prezzo massimo stimato = prezzo medio reale del ruolo del giocatore (dal listone) moltiplicato per il suo indice di Valore Reale (VR). Sotto, tra parentesi, la quotazione ufficiale corrente del listone.">
              Prezzo @if (sortColumn() === 'Prezzo_Massimo') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left hidden lg:table-cell" title="Valutazione Gruppo Esperti (forum.gruppoesperti.it), 1-5 stelle">Esperti</th>
          </tr>
        </thead>
        <tbody>
          @if (loading()) {
            @for (_ of skeletonRows; track $index) {
              <tr>
                @for (__ of [1,2,3,4,5,6,7,8,9,10,11,12]; track $index) {
                  <td class="px-3 py-2"><app-skeleton height="20px" /></td>
                }
              </tr>
            }
          } @else {
            @for (item of items(); track item.fantacalcio_id ?? item.id; let i = $index) {
              @let mp = mantraMap()[item.fantacalcio_id];
              <tr class="border-b cursor-pointer transition-colors"
                  style="border-color:var(--color-border)"
                  [style.background]="'transparent'"
                  (click)="playerSelected.emit(item)"
                  (mouseenter)="hoverId = item.id"
                  (mouseleave)="hoverId = null"
                  [style.backgroundColor]="hoverId === item.id ? 'var(--color-surface)' : 'transparent'">
                <td class="px-3 py-2.5 text-right font-mono text-xs"
                    style="color:var(--color-text-secondary)">{{ (page() - 1) * pageSize() + i + 1 }}</td>
                <td class="px-3 py-2.5 font-medium" style="color:var(--color-text-primary)">
                  {{ item.player_name }}
                </td>
                <td class="px-3 py-2.5 text-xs hidden sm:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.team ?? item.team_name ?? '—' }}
                  @if (teamStrength()[item.team ?? item.team_name]; as elo) {
                    <span class="elo-badge" [title]="'Elo: ' + (elo * 100 | number:'1.0-0') + '%'"
                          [style.opacity]="0.4 + elo * 0.6">●</span>
                  }
                </td>
                <td class="px-3 py-2.5 text-xs">
                  @if (mp?.ruolo_primario) {
                    <span class="font-medium" style="color:var(--color-text-primary)">{{ mp.ruolo_primario }}</span>
                  } @else {
                    <span class="italic opacity-50">—</span>
                  }
                </td>
                <td class="px-3 py-2.5 text-xs hidden lg:table-cell" style="color:var(--color-text-secondary)">
                  {{ item.stat_category ?? (mp?.ruoli_mantra?.join(', ') ?? '') }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-sm font-semibold"
                    style="color:var(--color-accent)">
                  {{ mp?.FP_Mantra != null ? (mp.FP_Mantra | number:'1.1-1') : '—' }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden md:table-cell"
                    style="color:var(--color-text-secondary)">
                  {{ mp?.VR != null ? (mp.VR | number:'1.0-0') : '—' }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs hidden md:table-cell"
                    style="color:var(--color-text-secondary)">
                  @let sp = mp?.start_probability;
                  {{ sp != null ? (sp * 100 | number:'1.0-0') + '%' : '—' }}
                </td>
                <td class="px-3 py-2.5 text-xs hidden md:table-cell">
                  @let mds = matchdayStatus()[item.fantacalcio_id];
                  @if (mds) {
                    <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                          [style.background]="(MATCHDAY_STATUS_CONFIG[mds.status]?.color ?? '#9CA3AF')">
                      {{ MATCHDAY_STATUS_CONFIG[mds.status]?.label ?? mds.status }}
                    </span>
                  } @else {
                    <span class="text-xs opacity-30">—</span>
                  }
                </td>
                <td class="px-3 py-2.5">
                  @if (mp && mp.Fase7) {
                    @let f7 = FASE7_LABELS[mp.Fase7];
                    <span class="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                          [style.background]="f7?.color ?? '#6B7280'"
                          [title]="FASE7_TOOLTIPS[mp.Fase7]">
                      <span class="sm:hidden">{{ f7?.icon ?? '' }} {{ mp.Fase7 }}</span>
                      <span class="hidden sm:inline">{{ f7?.icon ?? '' }} {{ f7?.label ?? mp.Fase7 }}</span>
                    </span>
                  } @else {
                    <span class="text-xs opacity-30" [title]="mp?.Fase7_Motivo ?? 'Nessuna categoria: il giocatore non rientra in nessuno dei 6 profili'">—</span>
                  }
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs whitespace-nowrap"
                    style="color:var(--color-text-secondary)">
                  {{ mp?.Prezzo_Massimo != null ? (mp.Prezzo_Massimo | number:'1.0-0') + ' cr' : '—' }}
                  @if (mp?.Pz1 != null) {
                    <div class="opacity-60" style="font-size:10px" title="Quotazione ufficiale corrente (listone)">list. {{ mp.Pz1 | number:'1.0-0' }}</div>
                  }
                </td>
                <td class="px-3 py-2.5 text-xs hidden lg:table-cell whitespace-nowrap">
                  @let er = expertRatings()[item.fantacalcio_id];
                  @if (er && er.rating != null) {
                    <span style="color:var(--color-accent)" [title]="er.comment ?? ''">{{ stars(er.rating) }}</span>
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
    .elo-badge { margin-left: 4px; color: var(--color-accent); font-size: 10px; }
  `],
})
export class PlayerTableComponent {
  readonly items = input.required<any[]>();
  readonly loading = input<boolean>(false);
  readonly page = input<number>(1);
  readonly pageSize = input<number>(50);
  readonly mantraMap = input<Record<number, MantraPlayer>>({});
  readonly matchdayStatus = input<Record<number, any>>({});
  readonly teamStrength = input<Record<string, number>>({});
  readonly expertRatings = input<Record<number, ExpertRatingWithFantacalcioId>>({});
  readonly sortColumn = input<string>('');
  readonly sortDirection = input<'asc' | 'desc'>('asc');
  readonly sortChanged = output<string>();
  readonly playerSelected = output<any>();

  hoverId: number | null = null;
  readonly skeletonRows = Array.from({ length: 8 });

  readonly stars = (rating: number): string =>
    '★'.repeat(rating) + '☆'.repeat(Math.max(0, 5 - rating));

  // Expose constants for template
  readonly FASE7_LABELS = FASE7_LABELS;
  readonly MATCHDAY_STATUS_CONFIG = MATCHDAY_STATUS_CONFIG;
  readonly FASE7_TOOLTIPS = FASE7_TOOLTIPS;
  /** Full legend of the 6 "Profilo" categories, shown on the column header's (i) icon. */
  readonly PROFILO_LEGEND = Object.values(FASE7_TOOLTIPS).join('\n');
}
