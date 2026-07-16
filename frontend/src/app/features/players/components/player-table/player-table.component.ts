import { Component, input, output } from '@angular/core';
import { PlayerSeasonStat } from '../../../../core/models/stats.models';
import { SkeletonComponent } from '../../../../shared/components/skeleton/skeleton.component';

@Component({
  selector: 'app-player-table',
  standalone: true,
  imports: [SkeletonComponent],
  template: `
    <div class="overflow-x-auto">
      <table class="w-full text-sm" style="border-collapse:collapse">
        <thead>
          <tr class="border-b text-xs font-medium uppercase tracking-wide"
              style="border-color:var(--color-border);color:var(--color-text-secondary)">
            <th class="px-3 py-2 text-right w-12">#</th>
            <th class="px-3 py-2 text-left">Player</th>
            <th class="px-3 py-2 text-left">Team</th>
            <th class="px-3 py-2 text-left">Category</th>
            <th class="px-3 py-2 text-right">Value</th>
            <th class="px-3 py-2 text-left">Season</th>
          </tr>
        </thead>
        <tbody>
          @if (loading()) {
            @for (_ of skeletonRows; track $index) {
              <tr>
                @for (__ of [1,2,3,4,5,6]; track $index) {
                  <td class="px-3 py-2"><app-skeleton height="20px" /></td>
                }
              </tr>
            }
          } @else {
            @for (item of items(); track item.id) {
              <tr class="border-b cursor-pointer transition-colors"
                  style="border-color:var(--color-border)"
                  [style.background]="'transparent'"
                  (click)="playerSelected.emit(item)"
                  (mouseenter)="hoverId = item.id"
                  (mouseleave)="hoverId = null"
                  [style.backgroundColor]="hoverId === item.id ? 'var(--color-surface)' : 'transparent'">
                <td class="px-3 py-2.5 text-right font-mono text-xs"
                    style="color:var(--color-text-secondary)">{{ item.rank ?? '—' }}</td>
                <td class="px-3 py-2.5 font-medium" style="color:var(--color-text-primary)">
                  {{ item.player_name }}
                </td>
                <td class="px-3 py-2.5" style="color:var(--color-text-secondary)">
                  {{ item.team_name ?? '—' }}
                </td>
                <td class="px-3 py-2.5" style="color:var(--color-text-secondary)">
                  {{ item.stat_category }}
                </td>
                <td class="px-3 py-2.5 text-right font-mono font-semibold"
                    style="color:var(--color-accent)">
                  {{ item.value ?? '—' }}
                </td>
                <td class="px-3 py-2.5 text-xs" style="color:var(--color-text-secondary)">
                  {{ item.season.season_label }}
                </td>
              </tr>
            }
          }
        </tbody>
      </table>
    </div>
  `,
  styles: [':host { display: block; }'],
})
export class PlayerTableComponent {
  readonly items = input.required<PlayerSeasonStat[]>();
  readonly loading = input<boolean>(false);
  readonly playerSelected = output<PlayerSeasonStat>();

  hoverId: number | null = null;
  readonly skeletonRows = Array.from({ length: 8 });
}
