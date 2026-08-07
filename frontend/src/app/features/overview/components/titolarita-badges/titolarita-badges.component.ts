import { Component, computed, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { MATCHDAY_STATUS_CONFIG } from '../../../../core/models/mantra.models';

/** Renders the 3 independent titolarità signals — real scraped status
 *  (probabili formazioni), ML probability, and Gruppo Esperti opinion —
 *  side by side, never merged into one. Single source of truth for both
 *  the compact table cluster (`size="sm"`) and the expanded drawer
 *  comparison (`size="lg"`), so the two views can never drift apart.
 *
 *  Mobile note (size="sm"): native `title` tooltips don't work on touch,
 *  so below the `sm` breakpoint each badge is a colour-only signal (dot /
 *  fill, no numeric text) — the full comparison lives in the drawer,
 *  opened by tapping the row, same "table scans, drawer explains" pattern
 *  already used across the app.
 */
@Component({
  selector: 'app-titolarita-badges',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    @if (size() === 'lg') {
      <div class="grid grid-cols-3 gap-2">
        <div class="rounded-lg px-2.5 py-2" style="background:var(--color-surface)">
          <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">Reale</p>
          @if (statusScraped(); as s) {
            <p class="text-sm font-semibold mt-0.5" [style.color]="statusColor()">
              {{ MATCHDAY_STATUS_CONFIG[s]?.label ?? s }}
            </p>
            @if (probabilityScraped() != null) {
              <p class="text-xs" style="color:var(--color-text-secondary)">{{ probabilityScraped() }}%</p>
            }
          } @else {
            <p class="text-sm mt-0.5 opacity-40">—</p>
          }
        </div>
        <div class="rounded-lg px-2.5 py-2" style="background:var(--color-surface)">
          <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">ML</p>
          @if (startProbability() != null) {
            <p class="text-sm font-semibold mt-0.5" style="color:var(--color-text-primary)">
              {{ startProbability()! * 100 | number:'1.0-0' }}%
            </p>
          } @else {
            <p class="text-sm mt-0.5 opacity-40">—</p>
          }
        </div>
        <div class="rounded-lg px-2.5 py-2" style="background:var(--color-surface)">
          <p class="text-[10px] uppercase tracking-wider" style="color:var(--color-text-secondary)">Esperti</p>
          @if (expertTitolarita() != null) {
            <p class="text-sm font-semibold mt-0.5" style="color:var(--color-text-primary)">
              {{ expertTitolarita() }}/10
            </p>
          } @else {
            <p class="text-sm mt-0.5 opacity-40">—</p>
          }
        </div>
      </div>
      @if (disagree()) {
        <p class="text-xs mt-2" style="color:#f59e0b">⚠️ Segnali di titolarità in disaccordo</p>
      }
    } @else {
      <div class="flex items-center gap-1.5">
        <span class="flex items-center gap-1" [title]="statusTooltip()">
          <span class="inline-block rounded-full" style="width:8px;height:8px" [style.background]="statusColor()"></span>
          @if (probabilityScraped() != null) {
            <span class="hidden sm:inline text-xs" style="color:var(--color-text-secondary)">{{ probabilityScraped() }}%</span>
          }
        </span>
        <span class="flex items-center gap-0.5" [title]="'Machine Learning — probabilità di titolarità' + (startProbability() != null ? ': ' + (startProbability()! * 100 | number:'1.0-0') + '%' : '')">
          <span class="text-[10px] font-medium opacity-60">ML</span>
          @if (startProbability() != null) {
            <span class="hidden sm:inline text-xs font-mono" style="color:var(--color-text-secondary)">
              {{ startProbability()! * 100 | number:'1.0-0' }}%
            </span>
          } @else {
            <span class="text-xs opacity-30">—</span>
          }
        </span>
        <span class="flex items-center gap-0.5" [title]="'Gruppo Esperti — titolarità prevista' + (expertTitolarita() != null ? ': ' + expertTitolarita() + '/10' : '')">
          <span class="text-[10px] font-medium opacity-60">GE</span>
          @if (expertTitolarita() != null) {
            <span class="hidden sm:inline text-xs font-mono" style="color:var(--color-text-secondary)">
              {{ expertTitolarita() }}/10
            </span>
          } @else {
            <span class="text-xs opacity-30">—</span>
          }
        </span>
      </div>
    }
  `,
})
export class TitolaritaBadgesComponent {
  readonly statusScraped = input<string | null>(null);
  readonly probabilityScraped = input<number | null>(null);
  readonly startProbability = input<number | null>(null);
  readonly expertTitolarita = input<number | null>(null);
  readonly size = input<'sm' | 'lg'>('sm');

  readonly MATCHDAY_STATUS_CONFIG = MATCHDAY_STATUS_CONFIG;

  readonly statusColor = computed(() => {
    const s = this.statusScraped();
    return s ? (MATCHDAY_STATUS_CONFIG[s]?.color ?? '#9CA3AF') : '#9CA3AF33';
  });

  readonly statusTooltip = computed(() => {
    const s = this.statusScraped();
    if (!s) return 'Probabili formazioni — nessun dato';
    const label = MATCHDAY_STATUS_CONFIG[s]?.label ?? s;
    const pct = this.probabilityScraped();
    return `Probabili formazioni: ${label}` + (pct != null ? ` (${pct}%)` : '');
  });

  /** Simple disagreement heuristic: normalize the 3 available signals to
   *  0-100 and flag when the spread between them exceeds 40 points. */
  readonly disagree = computed(() => {
    const values: number[] = [];
    const p = this.probabilityScraped();
    if (p != null) values.push(p);
    const sp = this.startProbability();
    if (sp != null) values.push(sp * 100);
    const et = this.expertTitolarita();
    if (et != null) values.push(et * 10);
    if (values.length < 2) return false;
    return Math.max(...values) - Math.min(...values) > 40;
  });
}
