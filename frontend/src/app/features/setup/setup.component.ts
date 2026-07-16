import { Component, inject, signal } from '@angular/core';
import { Router } from '@angular/router';

const API_KEY_STORAGE_KEY = 'fanta_api_key';

@Component({
  selector: 'app-setup',
  standalone: true,
  template: `
    <div class="min-h-screen flex items-center justify-center p-6"
         style="background: var(--color-bg)">
      <div class="w-full max-w-md card space-y-6">
        <div class="text-center">
          <div class="mx-auto mb-4 h-12 w-12 rounded-xl bg-brand-500 flex items-center justify-center">
            <span class="text-white text-xl font-bold">FI</span>
          </div>
          <h1 class="text-xl font-semibold" style="color: var(--color-text-primary)">
            FantaIntelligence
          </h1>
          <p class="mt-1 text-sm" style="color: var(--color-text-secondary)">
            Enter your API key to continue
          </p>
        </div>

        <form (submit)="onSubmit($event)" class="space-y-4">
          <div>
            <label for="api-key" class="block text-sm font-medium mb-1.5"
                   style="color: var(--color-text-secondary)">
              API Key
            </label>
            <input
              id="api-key"
              type="password"
              autocomplete="current-password"
              class="w-full rounded-lg border px-3 py-2 text-sm font-mono outline-none
                     focus:ring-2 focus:ring-brand-500/50 transition"
              style="background: var(--color-surface); border-color: var(--color-border);
                     color: var(--color-text-primary)"
              placeholder="sk-..."
              [value]="apiKey()"
              (input)="apiKey.set($any($event.target).value)"
            />
          </div>

          <button
            type="submit"
            class="w-full rounded-lg bg-brand-500 px-4 py-2.5 text-sm font-semibold
                   text-white transition hover:bg-brand-600 focus:outline-none
                   focus:ring-2 focus:ring-brand-500/50 disabled:opacity-50"
            [disabled]="!apiKey().trim()"
          >
            Continue to Dashboard
          </button>
        </form>
      </div>
    </div>
  `,
})
export class SetupComponent {
  readonly apiKey = signal('');
  private readonly router = inject(Router);

  onSubmit(event: Event): void {
    event.preventDefault();
    const key = this.apiKey().trim();
    if (!key) return;
    localStorage.setItem(API_KEY_STORAGE_KEY, key);
    this.router.navigate(['/dashboard']);
  }
}
