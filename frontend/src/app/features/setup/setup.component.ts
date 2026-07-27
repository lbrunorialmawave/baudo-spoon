import { Component, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

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
            Accedi per continuare
          </p>
        </div>

        <form (submit)="onSubmit($event)" class="space-y-4">
          <div>
            <label for="email" class="block text-sm font-medium mb-1.5"
                   style="color: var(--color-text-secondary)">
              Email
            </label>
            <input
              id="email"
              type="email"
              autocomplete="email"
              class="w-full rounded-lg border px-3 py-2 text-sm outline-none
                     focus:ring-2 focus:ring-brand-500/50 transition"
              style="background: var(--color-surface); border-color: var(--color-border);
                     color: var(--color-text-primary)"
              placeholder="you@example.com"
              [value]="email()"
              (input)="email.set($any($event.target).value)"
            />
          </div>

          <div>
            <label for="password" class="block text-sm font-medium mb-1.5"
                   style="color: var(--color-text-secondary)">
              Password
            </label>
            <input
              id="password"
              type="password"
              autocomplete="current-password"
              class="w-full rounded-lg border px-3 py-2 text-sm outline-none
                     focus:ring-2 focus:ring-brand-500/50 transition"
              style="background: var(--color-surface); border-color: var(--color-border);
                     color: var(--color-text-primary)"
              placeholder="••••••••"
              [value]="password()"
              (input)="password.set($any($event.target).value)"
            />
          </div>

          @if (error()) {
            <p class="text-sm" style="color: var(--color-danger, #ef4444)">
              {{ error() }}
            </p>
          }

          <button
            type="submit"
            class="w-full rounded-lg bg-brand-500 px-4 py-2.5 text-sm font-semibold
                   text-white transition hover:bg-brand-600 focus:outline-none
                   focus:ring-2 focus:ring-brand-500/50 disabled:opacity-50"
            [disabled]="loading() || !email().trim() || !password().trim()"
          >
            {{ loading() ? 'Accesso in corso…' : 'Accedi' }}
          </button>
        </form>
      </div>
    </div>
  `,
})
export class SetupComponent {
  readonly email = signal('');
  readonly password = signal('');
  readonly loading = signal(false);
  readonly error = signal('');

  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);

  onSubmit(event: Event): void {
    event.preventDefault();
    if (!this.email().trim() || !this.password().trim()) return;

    this.loading.set(true);
    this.error.set('');

    this.auth.login(this.email().trim(), this.password().trim()).subscribe({
      next: () => this.router.navigate(['/dashboard']),
      error: () => {
        this.error.set('Email o password errati');
        this.loading.set(false);
      },
    });
  }
}
