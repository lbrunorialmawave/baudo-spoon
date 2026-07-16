import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-not-found',
  standalone: true,
  imports: [RouterLink],
  template: `
    <div class="min-h-screen flex items-center justify-center p-6"
         style="background: var(--color-bg)">
      <div class="text-center space-y-4">
        <p class="text-7xl font-black text-brand-500">404</p>
        <h1 class="text-xl font-semibold" style="color: var(--color-text-primary)">
          Page not found
        </h1>
        <p class="text-sm" style="color: var(--color-text-secondary)">
          The route you requested doesn't exist.
        </p>
        <a routerLink="/dashboard"
           class="inline-block mt-2 rounded-lg bg-brand-500 px-4 py-2 text-sm
                  font-semibold text-white hover:bg-brand-600 transition">
          Back to Dashboard
        </a>
      </div>
    </div>
  `,
})
export class NotFoundComponent {}
