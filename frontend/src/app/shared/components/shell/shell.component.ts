import { Component, signal, afterRenderEffect, inject, PLATFORM_ID } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { isPlatformBrowser } from '@angular/common';

const NAV_ITEMS = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`,
  },
  {
    path: '/players',
    label: 'Players',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path stroke-linecap="round" d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>`,
  },
  {
    path: '/quotations',
    label: 'Quotations',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M7 7h10M7 12h6m-6 5h4"/><rect x="3" y="3" width="18" height="18" rx="2"/></svg>`,
  },
  {
    path: '/matches',
    label: 'Matches',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="17" rx="2"/><path stroke-linecap="round" d="M3 9h18M8 2v4M16 2v4"/></svg>`,
  },
  {
    path: '/id-mapping',
    label: 'ID Mapping',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M8 7h12M8 12h12M8 17h12M4 7h.01M4 12h.01M4 17h.01"/></svg>`,
  },
  {
    path: '/predictions',
    label: 'Predictions',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M3 17l5-5 4 4 9-10"/></svg>`,
  },
  {
    path: '/optimizer',
    label: 'Optimizer',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>`,
  },
  {
    path: '/auction',
    label: 'Auction',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>`,
  },
  {
    path: '/model-monitoring',
    label: 'Model Monitoring',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>`,
  },
  {
    path: '/admin',
    label: 'Admin',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path stroke-linecap="round" d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>`,
  },
] as const;

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="app-shell" [class.collapsed]="collapsed()">
      <!-- ── Sidebar ──────────────────────────────────────── -->
      <aside class="sidebar">
        <!-- Logo -->
        <div class="sidebar-logo">
          <div class="logo-mark">
            <span>FI</span>
          </div>
          @if (!collapsed()) {
            <span class="logo-text">FantaIntelligence</span>
          }
        </div>

        <!-- Toggle -->
        <button
          class="collapse-btn"
          (click)="collapsed.update(v => !v)"
          [attr.aria-label]="collapsed() ? 'Expand sidebar' : 'Collapse sidebar'"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="1.8"
            [style.transform]="collapsed() ? 'rotate(180deg)' : 'none'"
            style="transition:transform 150ms ease"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        <!-- Nav -->
        <nav class="sidebar-nav">
          @for (item of navItems; track item.path) {
            <a
              [routerLink]="item.path"
              routerLinkActive="nav-link--active"
              [routerLinkActiveOptions]="{ exact: false }"
              class="nav-link"
              [attr.title]="collapsed() ? item.label : null"
            >
              <span class="nav-icon" [innerHTML]="item.icon"></span>
              @if (!collapsed()) {
                <span class="nav-label">{{ item.label }}</span>
              }
            </a>
          }
        </nav>

        <!-- API key status -->
        <div class="sidebar-footer">
          <div class="status-dot" [class.status-dot--ok]="apiKeyPresent()"></div>
          @if (!collapsed()) {
            <span class="status-label">
              {{ apiKeyPresent() ? 'API Key Set' : 'No API Key' }}
            </span>
          }
        </div>
      </aside>

      <!-- ── Main ─────────────────────────────────────────── -->
      <main class="shell-main">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [
    `
      .app-shell {
        display: grid;
        grid-template-columns: 240px 1fr;
        height: 100dvh;
        overflow: hidden;
        transition: grid-template-columns 150ms ease;
      }
      .app-shell.collapsed {
        grid-template-columns: 60px 1fr;
      }
      .sidebar {
        display: flex;
        flex-direction: column;
        border-right: 1px solid var(--color-border);
        background: var(--color-surface);
        overflow: hidden;
        position: relative;
      }
      .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 16px 14px 12px;
        border-bottom: 1px solid var(--color-border);
        min-height: 56px;
      }
      .logo-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: var(--color-accent);
        flex-shrink: 0;
      }
      .logo-mark span {
        font-size: 12px;
        font-weight: 700;
        color: #fff;
      }
      .logo-text {
        font-size: 13px;
        font-weight: 600;
        color: var(--color-text-primary);
        white-space: nowrap;
        overflow: hidden;
      }
      .collapse-btn {
        position: absolute;
        right: -12px;
        top: 66px;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        border: 1px solid var(--color-border);
        background: var(--color-surface-raised);
        color: var(--color-text-secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 10;
      }
      .collapse-btn:hover {
        color: var(--color-text-primary);
      }
      .sidebar-nav {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding: 12px 8px;
        overflow-y: auto;
      }
      .nav-link {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 8px;
        text-decoration: none;
        color: var(--color-text-secondary);
        font-size: 13px;
        font-weight: 500;
        transition:
          background 120ms,
          color 120ms;
        white-space: nowrap;
      }
      .nav-link:hover {
        background: var(--color-surface-raised);
        color: var(--color-text-primary);
      }
      .nav-link--active {
        background: color-mix(in srgb, var(--color-accent) 12%, transparent);
        color: var(--color-accent);
      }
      .nav-icon {
        display: flex;
        align-items: center;
        flex-shrink: 0;
      }
      .nav-label {
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .sidebar-footer {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 14px;
        border-top: 1px solid var(--color-border);
        min-height: 48px;
      }
      .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #f59e0b;
        flex-shrink: 0;
      }
      .status-dot--ok {
        background: #22c55e;
      }
      .status-label {
        font-size: 12px;
        color: var(--color-text-secondary);
        white-space: nowrap;
        overflow: hidden;
      }
      .shell-main {
        overflow-y: auto;
        background: var(--color-bg);
      }
    `,
  ],
})
export class ShellComponent {
  readonly navItems = NAV_ITEMS;
  readonly collapsed = signal(false);
  readonly apiKeyPresent = signal(false);

  private readonly platformId = inject(PLATFORM_ID);

  constructor() {
    afterRenderEffect(() => {
      if (isPlatformBrowser(this.platformId)) {
        this.apiKeyPresent.set(!!localStorage.getItem('fanta_api_key'));
      }
    });
  }
}
