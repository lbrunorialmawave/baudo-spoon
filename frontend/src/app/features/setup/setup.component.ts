import { Component, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { AuthService } from '../../core/services/auth.service';

type Mode = 'login' | 'register';

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
            {{ mode() === 'login' ? 'Accedi per continuare' : 'Crea un nuovo account' }}
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
              [attr.autocomplete]="mode() === 'login' ? 'current-password' : 'new-password'"
              class="w-full rounded-lg border px-3 py-2 text-sm outline-none
                     focus:ring-2 focus:ring-brand-500/50 transition"
              style="background: var(--color-surface); border-color: var(--color-border);
                     color: var(--color-text-primary)"
              placeholder="••••••••"
              [value]="password()"
              (input)="password.set($any($event.target).value)"
            />
          </div>

          @if (mode() === 'register') {
            <div>
              <label for="confirmPassword" class="block text-sm font-medium mb-1.5"
                     style="color: var(--color-text-secondary)">
                Conferma password
              </label>
              <input
                id="confirmPassword"
                type="password"
                autocomplete="new-password"
                class="w-full rounded-lg border px-3 py-2 text-sm outline-none
                       focus:ring-2 focus:ring-brand-500/50 transition"
                style="background: var(--color-surface); border-color: var(--color-border);
                       color: var(--color-text-primary)"
                placeholder="••••••••"
                [value]="confirmPassword()"
                (input)="confirmPassword.set($any($event.target).value)"
              />
            </div>
          }

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
            [disabled]="loading() || !canSubmit()"
          >
            {{ submitLabel() }}
          </button>
        </form>

        <p class="text-center text-sm" style="color: var(--color-text-secondary)">
          @if (mode() === 'login') {
            Non hai un account?
            <button type="button" class="font-semibold text-brand-500 hover:underline" (click)="toggleMode()">
              Registrati
            </button>
          } @else {
            Hai già un account?
            <button type="button" class="font-semibold text-brand-500 hover:underline" (click)="toggleMode()">
              Accedi
            </button>
          }
        </p>
      </div>
    </div>
  `,
})
export class SetupComponent {
  readonly mode = signal<Mode>('login');
  readonly email = signal('');
  readonly password = signal('');
  readonly confirmPassword = signal('');
  readonly loading = signal(false);
  readonly error = signal('');

  readonly canSubmit = computed(() => {
    if (!this.email().trim() || !this.password().trim()) return false;
    if (this.mode() === 'register' && !this.confirmPassword().trim()) return false;
    return true;
  });

  readonly submitLabel = computed(() => {
    if (this.loading()) return this.mode() === 'login' ? 'Accesso in corso…' : 'Registrazione in corso…';
    return this.mode() === 'login' ? 'Accedi' : 'Registrati';
  });

  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);

  toggleMode(): void {
    this.mode.set(this.mode() === 'login' ? 'register' : 'login');
    this.error.set('');
    this.confirmPassword.set('');
  }

  onSubmit(event: Event): void {
    event.preventDefault();
    if (!this.canSubmit()) return;

    const email = this.email().trim();
    const password = this.password().trim();

    if (this.mode() === 'register' && password !== this.confirmPassword().trim()) {
      this.error.set('Le password non coincidono');
      return;
    }

    this.loading.set(true);
    this.error.set('');

    const request$ = this.mode() === 'login'
      ? this.auth.login(email, password)
      : this.auth.register(email, password);

    request$.subscribe({
      next: () => this.router.navigate(['/dashboard']),
      error: (err: HttpErrorResponse) => {
        if (this.mode() === 'register') {
          this.error.set(err.status === 409 ? 'Email già registrata' : 'Registrazione non riuscita');
        } else {
          this.error.set('Email o password errati');
        }
        this.loading.set(false);
      },
    });
  }
}

