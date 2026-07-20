import { Component, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { MantraPlayer, FASE7_LABELS, MATCHDAY_STATUS_CONFIG } from '../../../../core/models/mantra.models';
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
            <th class="px-3 py-2 text-right sortable hidden md:table-cell" (click)="sortChanged.emit('VR')" title="Voto Ricevuto — media dei voti in pagella (scala 0-100)">
              VR @if (sortColumn() === 'VR') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
            <th class="px-3 py-2 text-left hidden md:table-cell" title="Stato per la prossima giornata (infortunato, squalificato, etc.)">Status</th>
            <th class="px-3 py-2 text-left" title="Classificazione Fase 7 del calciatore (TOP, AFFARE, CERTEZZA, SCOMMESSA, SOPRAVALUTATO, GIUSTO)">Fase 7</th>
            <th class="px-3 py-2 text-right sortable" (click)="sortChanged.emit('Prezzo_Massimo')" title="Prezzo massimo di mercato stimato in crediti">
              Prezzo @if (sortColumn() === 'Prezzo_Massimo') { <span style="color:var(--color-accent)">{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span> }
            </th>
          </tr>
        </thead>
        <tbody>
          @if (loading()) {
            @for (_ of skeletonRows; track $index) {
              <tr>
                @for (__ of [1,2,3,4,5,6,7,8,9,10]; track $index) {
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
                    <span class="text-xs opacity-30" title="Nessuna classificazione Fase 7 — il giocatore non rientra in nessuna categoria (dati insufficienti o profilo nella media non classificabile)">—</span>
                  }
                </td>
                <td class="px-3 py-2.5 text-right font-mono text-xs whitespace-nowrap"
                    style="color:var(--color-text-secondary)">
                  {{ mp?.Pz1 != null ? (mp.Pz1 | number:'1.0-0') + ' cr' : '—' }}
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
export class PlayerTableComponent {
  readonly items = input.required<any[]>();
  readonly loading = input<boolean>(false);
  readonly page = input<number>(1);
  readonly pageSize = input<number>(50);
  readonly mantraMap = input<Record<number, MantraPlayer>>({});
  readonly matchdayStatus = input<Record<number, any>>({});
  readonly sortColumn = input<string>('');
  readonly sortDirection = input<'asc' | 'desc'>('asc');
  readonly sortChanged = output<string>();
  readonly playerSelected = output<any>();

  hoverId: number | null = null;
  readonly skeletonRows = Array.from({ length: 8 });

  // Expose constants for template
  readonly FASE7_LABELS = FASE7_LABELS;
  readonly MATCHDAY_STATUS_CONFIG = MATCHDAY_STATUS_CONFIG;
  readonly FASE7_TOOLTIPS: Record<string, string> = {
    TOP: '🏆 TOP — Giocatore d\'élite: FP alto e VR bilanciato. Investimento sicuro.',
    AFFARE: '💎 AFFARE — Sottovalutato dal mercato: FP alto, prezzo basso. Ottimo rapporto Q/P.',
    SCOMMESSA: '🔄 SCOMMESSA — Potenziale inespresso: FP basso ma VR alto. Può esplodere.',
    CERTEZZA: '✅ CERTEZZA — Rendimento stabile e affidabile. Poche sorprese.',
    SOPRAVALUTATO: '⚠️ SOPRAVALUTATO — Prezzo gonfiato: VR basso rispetto al FP. Rischi.',
    GIUSTO: '⚖️ GIUSTO — Nella media: FP e VR allineati al prezzo di mercato.',
  };
}
