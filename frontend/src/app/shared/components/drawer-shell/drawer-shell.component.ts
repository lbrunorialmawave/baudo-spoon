import { Component, input, output } from '@angular/core';

/** Shared backdrop + slide-in panel chrome for detail drawers. Same
 *  proven markup/CSS as PlayerDrawerComponent / PredictionDrawerComponent
 *  (full-width below 640px, fixed 400px above), extracted so new drawers
 *  don't duplicate it. The two existing drawers keep their own copy for
 *  now — migrating them is a separate, lower-risk follow-up. */
@Component({
  selector: 'app-drawer-shell',
  standalone: true,
  template: `
    <div class="drawer-backdrop" (click)="closed.emit()"></div>

    <aside class="drawer-panel">
      <div class="drawer-header">
        <div class="min-w-0">
          <h2 class="truncate font-semibold" style="color:var(--color-text-primary)">
            {{ title() }}
          </h2>
          @if (subtitle()) {
            <p class="text-xs mt-0.5" style="color:var(--color-text-secondary)">{{ subtitle() }}</p>
          }
        </div>
        <button class="close-btn" (click)="closed.emit()" aria-label="Close">✕</button>
      </div>

      <div class="drawer-body">
        <ng-content />
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
      width: 100vw;
      display: flex; flex-direction: column;
      background: var(--color-surface);
      border-left: 1px solid var(--color-border);
      animation: slide-in 180ms ease-out;
      padding-bottom: env(safe-area-inset-bottom, 0);
      padding-right: env(safe-area-inset-right, 0);
    }
    @media (min-width: 640px) {
      .drawer-panel { width: 400px; }
    }
    @media (prefers-reduced-motion: reduce) {
      .drawer-panel { animation: none; }
    }
    @keyframes slide-in {
      from { transform: translateX(100%); }
      to   { transform: translateX(0); }
    }
    .drawer-header {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
      padding: 16px; border-bottom: 1px solid var(--color-border);
      padding-top: max(16px, env(safe-area-inset-top, 0));
    }
    .close-btn {
      flex-shrink: 0; width: 44px; height: 44px;
      border-radius: 8px; font-size: 12px;
      background: var(--color-surface-raised);
      color: var(--color-text-secondary);
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
    }
    .close-btn:hover { color: var(--color-text-primary); }
    .drawer-body { flex: 1; overflow-y: auto; padding: 16px; }
  `],
})
export class DrawerShellComponent {
  readonly title = input.required<string>();
  readonly subtitle = input<string | null>(null);
  readonly closed = output<void>();
}
