import { Component, input } from '@angular/core';

/** RFC 7807 Problem Details error display */
@Component({
  selector: 'app-error-boundary',
  standalone: true,
  template: `
    <div
      class="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4"
      role="alert"
      aria-live="assertive"
    >
      <div class="flex items-start gap-3">
        <svg class="mt-0.5 h-5 w-5 shrink-0 text-rose-400"
             xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2"
             aria-hidden="true">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <div>
          @if (title()) {
            <p class="text-sm font-semibold text-rose-300">{{ title() }}</p>
          }
          <p class="text-sm text-rose-400">{{ message() }}</p>
        </div>
      </div>
    </div>
  `,
})
export class ErrorBoundaryComponent {
  readonly message = input.required<string>();
  readonly title = input<string>('');
}
