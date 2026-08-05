import { Component } from '@angular/core';

/** Full-screen splash shown while the backend wakes up from a Render cold start. */
@Component({
  selector: 'app-health-gate',
  standalone: true,
  template: `
    <div
      class="flex min-h-screen w-full flex-col items-center justify-center gap-4 bg-slate-950 px-6 text-center"
      role="status"
      aria-live="polite"
    >
      <svg
        class="h-8 w-8 animate-spin text-sky-400"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
      >
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path
          class="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4Z"
        />
      </svg>
      <p class="text-sm font-medium text-slate-200">Il server si sta risvegliando…</p>
      <p class="max-w-xs text-xs text-slate-400">
        Il piano gratuito si mette in pausa quando è inattivo. Può richiedere fino a un minuto.
      </p>
    </div>
  `,
})
export class HealthGateComponent {}
