import { Component, computed, input, output } from '@angular/core';
import { MatchStat } from '../../../../core/models/stats.models';

@Component({
  selector: 'app-match-detail',
  standalone: true,
  template: `
    <div class="drawer-backdrop" (click)="closed.emit()"></div>
    <aside class="drawer-panel">
      <!-- Header -->
      <div class="drawer-header">
        <div class="min-w-0">
          <h2 class="truncate font-semibold" style="color:var(--color-text-primary)">
            {{ match().match_name }}
          </h2>
          <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">
            {{ match().match_date ?? '—' }}
            @if (match().score) { · {{ match().score }} }
            @if (match().round_num) { · Round {{ match().round_num }} }
          </p>
        </div>
        <button class="close-btn" (click)="closed.emit()" aria-label="Close">✕</button>
      </div>

      <div class="drawer-body">
        <!-- Match summary -->
        <div class="grid grid-cols-3 gap-3 mb-5">
          <div class="rounded-lg p-3 text-center" style="background:var(--color-surface)">
            <p class="text-xs" style="color:var(--color-text-secondary)">Score</p>
            <p class="font-bold mt-0.5" style="color:var(--color-text-primary)">
              {{ match().score ?? '—' }}
            </p>
          </div>
          <div class="rounded-lg p-3 text-center" style="background:var(--color-surface)">
            <p class="text-xs" style="color:var(--color-text-secondary)">Points</p>
            <p class="font-bold mt-0.5" style="color:var(--color-accent)">
              {{ match().points ?? '—' }}
            </p>
          </div>
          <div class="rounded-lg p-3 text-center" style="background:var(--color-surface)">
            <p class="text-xs" style="color:var(--color-text-secondary)">Side</p>
            <p class="font-bold mt-0.5" style="color:var(--color-text-primary)">
              {{ match().side ?? '—' }}
            </p>
          </div>
        </div>

        <!-- Raw stats JSONB -->
        @if (statsEntries().length) {
          <h3 class="section-title">Match Stats</h3>
          <div class="space-y-1">
            @for (entry of statsEntries(); track entry[0]) {
              <div class="flex items-center justify-between rounded-lg px-3 py-2"
                   style="background:var(--color-surface)">
                <span class="text-xs capitalize" style="color:var(--color-text-secondary)">
                  {{ entry[0].replace(/_/g, ' ') }}
                </span>
                <span class="text-xs font-mono font-medium" style="color:var(--color-text-primary)">
                  {{ formatVal(entry[1]) }}
                </span>
              </div>
            }
          </div>
        }
      </div>
    </aside>
  `,
  styles: [`
    :host { display: contents; }
    .drawer-backdrop {
      position: fixed; inset: 0; z-index: 40;
      background: rgba(0,0,0,0.5);
    }
    .drawer-panel {
      position: fixed; right: 0; top: 0; bottom: 0; z-index: 50;
      width: 420px;
      display: flex; flex-direction: column;
      background: var(--color-surface);
      border-left: 1px solid var(--color-border);
      animation: slide-in 180ms ease-out;
    }
    @keyframes slide-in {
      from { transform: translateX(100%); }
      to   { transform: translateX(0); }
    }
    .drawer-header {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
      padding: 16px; border-bottom: 1px solid var(--color-border);
    }
    .close-btn {
      flex-shrink: 0; width: 28px; height: 28px;
      border-radius: 6px; font-size: 12px;
      background: var(--color-surface-raised);
      color: var(--color-text-secondary);
      display: flex; align-items: center; justify-content: center; cursor: pointer;
    }
    .close-btn:hover { color: var(--color-text-primary); }
    .drawer-body { flex: 1; overflow-y: auto; padding: 16px; }
    .section-title {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.06em; margin-bottom: 8px;
      color: var(--color-text-secondary);
    }
  `],
})
export class MatchDetailComponent {
  readonly match = input.required<MatchStat>();
  readonly closed = output<void>();

  readonly statsEntries = computed(() => Object.entries(this.match().stats));

  formatVal(v: unknown): string {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }
}
