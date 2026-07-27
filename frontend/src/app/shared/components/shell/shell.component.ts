import { Component, signal, afterRenderEffect, inject, PLATFORM_ID, computed, HostListener } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { isPlatformBrowser } from '@angular/common';
import { AuthService } from '../../../core/services/auth.service';

interface NavItem {
  path: string;
  label: string;
  shortLabel: string;
  icon: string;
}

const NAV_ITEMS: readonly NavItem[] = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    shortLabel: 'Home',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`,
  },
  {
    path: '/players',
    label: 'Players',
    shortLabel: 'Players',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path stroke-linecap="round" d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>`,
  },
  {
    path: '/quotations',
    label: 'Quotations',
    shortLabel: 'Quote',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M7 7h10M7 12h6m-6 5h4"/><rect x="3" y="3" width="18" height="18" rx="2"/></svg>`,
  },
  {
    path: '/matches',
    label: 'Matches',
    shortLabel: 'Matches',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="17" rx="2"/><path stroke-linecap="round" d="M3 9h18M8 2v4M16 2v4"/></svg>`,
  },
  {
    path: '/predictions',
    label: 'Predictions',
    shortLabel: 'Predict',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M3 17l5-5 4 4 9-10"/></svg>`,
  },
  {
    path: '/optimizer',
    label: 'Optimizer',
    shortLabel: 'Optim',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>`,
  },
  {
    path: '/auction',
    label: 'Auction',
    shortLabel: 'Auction',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>`,
  },
  {
    path: '/id-mapping',
    label: 'ID Mapping',
    shortLabel: 'IDs',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M8 7h12M8 12h12M8 17h12M4 7h.01M4 12h.01M4 17h.01"/></svg>`,
  },
  {
    path: '/model-monitoring',
    label: 'Model Monitoring',
    shortLabel: 'Models',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M3 3v18h18M7 14l4-4 4 4 5-6"/></svg>`,
  },
  {
    path: '/admin',
    label: 'Admin',
    shortLabel: 'Admin',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path stroke-linecap="round" d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>`,
  },
];

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="app-shell" [class.collapsed]="collapsed()">
      <!-- ── Sidebar (desktop ≥ lg) ────────────────────────── -->
      <aside class="sidebar" aria-label="Primary navigation">
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
          (click)="toggleCollapsed()"
          [attr.aria-label]="collapsed() ? 'Expand sidebar' : 'Collapse sidebar'"
          [attr.aria-expanded]="!collapsed()"
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
          @for (item of navItems(); track item.path) {
            <a
              [routerLink]="item.path"
              routerLinkActive="nav-link--active"
              [routerLinkActiveOptions]="{ exact: false }"
              class="nav-link"
              [attr.title]="collapsed() ? item.label : null"
              [attr.aria-label]="item.label"
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
          <div class="status-dot" [class.status-dot--ok]="isAuthenticated()"></div>
          @if (!collapsed()) {
            <span class="status-label">
              {{ isAuthenticated() ? (isAdmin() ? 'Admin' : 'Member') : 'Not logged in' }}
            </span>
          }
        </div>
      </aside>

      <!-- ── Main ─────────────────────────────────────────── -->
      <main class="shell-main" id="main-content">
        <router-outlet />
      </main>

      <!-- ── Bottom nav (mobile < lg) ─────────────────────── -->
      <nav class="bottom-nav" aria-label="Primary navigation (mobile)">
        <div class="bottom-nav-inner">
          @for (item of mobileNavItems(); track item.path) {
            <a
              [routerLink]="item.path"
              routerLinkActive="bottom-nav-link--active"
              [routerLinkActiveOptions]="{ exact: false }"
              class="bottom-nav-link"
              [attr.aria-label]="item.label"
            >
              <span class="bottom-nav-icon" [innerHTML]="item.icon"></span>
              <span class="bottom-nav-label">{{ item.shortLabel }}</span>
            </a>
          }
          <button
            class="bottom-nav-link bottom-nav-link--menu"
            (click)="toggleMenu()"
            [attr.aria-expanded]="menuOpen()"
            aria-controls="mobile-menu"
            [attr.aria-label]="menuOpen() ? 'Close menu' : 'Open menu'"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
              @if (menuOpen()) {
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
              } @else {
                <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16" />
              }
            </svg>
            <span class="bottom-nav-label">{{ menuOpen() ? 'Close' : 'More' }}</span>
          </button>
        </div>
      </nav>

      <!-- ── Mobile menu drawer ───────────────────────────── -->
      @if (menuOpen()) {
        <div class="mobile-menu-backdrop" (click)="closeMenu()" aria-hidden="true"></div>
        <div id="mobile-menu" class="mobile-menu" role="dialog" aria-modal="true" aria-label="Full navigation">
          <div class="mobile-menu-header">
            <div class="logo-mark">
              <span>FI</span>
            </div>
            <span class="logo-text">FantaIntelligence</span>
            <button class="mobile-menu-close" (click)="closeMenu()" aria-label="Close menu">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <nav class="mobile-menu-nav">
            @for (item of navItems(); track item.path) {
              <a
                [routerLink]="item.path"
                routerLinkActive="mobile-menu-link--active"
                [routerLinkActiveOptions]="{ exact: false }"
                class="mobile-menu-link"
                (click)="closeMenu()"
              >
                <span class="mobile-menu-icon" [innerHTML]="item.icon"></span>
                <span>{{ item.label }}</span>
              </a>
            }
          </nav>
          <div class="mobile-menu-footer">
            <div class="status-dot" [class.status-dot--ok]="isAuthenticated()"></div>
            <span class="status-label">
              {{ isAuthenticated() ? (isAdmin() ? 'Admin' : 'Member') : 'Not logged in' }}
            </span>
          </div>
        </div>
      }
    </div>
  `,
  styles: [
    `
      :host {
        display: block;
        min-height: 100dvh;
      }

      .app-shell {
        display: grid;
        grid-template-columns: 1fr;
        grid-template-rows: 1fr auto;
        min-height: 100dvh;
        overflow: hidden;
        background: var(--color-bg);
      }

      /* ── Sidebar: hidden on mobile, visible from lg ── */
      .sidebar {
        display: none;
        flex-direction: column;
        background: var(--color-surface);
        overflow: hidden;
        position: relative;
        height: 100dvh;
        position: sticky;
        top: 0;
      }

      @media (min-width: 64rem /* 1024px — Tailwind lg */) {
        .app-shell {
          grid-template-columns: auto 1fr;
          grid-template-rows: 1fr;
        }
        .app-shell.collapsed {
          grid-template-columns: 60px 1fr;
        }
        .sidebar {
          display: flex;
        }
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
        text-overflow: ellipsis;
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
        min-height: 0;
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
        min-height: 40px;
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
        text-overflow: ellipsis;
      }

      /* ── Main: scroll container ─────────────────────── */
      .shell-main {
        overflow-y: auto;
        overflow-x: hidden;
        background: var(--color-bg);
        min-height: 0;
        -webkit-overflow-scrolling: touch;
      }

      /* ── Bottom nav: mobile only ───────────────────── */
      .bottom-nav {
        display: block;
        background: var(--color-surface);
        border-top: 1px solid var(--color-border);
        padding-bottom: env(safe-area-inset-bottom, 0);
        z-index: 30;
        position: sticky;
        bottom: 0;
      }
      @media (min-width: 64rem) {
        .bottom-nav {
          display: none;
        }
      }
      .bottom-nav-inner {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 2px;
        padding: 4px 4px 6px;
      }
      .bottom-nav-link {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 2px;
        padding: 6px 2px 4px;
        border-radius: 8px;
        text-decoration: none;
        color: var(--color-text-secondary);
        background: transparent;
        border: 0;
        cursor: pointer;
        font: inherit;
        min-height: 48px;
        transition: color 120ms;
      }
      .bottom-nav-link:hover,
      .bottom-nav-link:focus-visible {
        color: var(--color-text-primary);
      }
      .bottom-nav-link--active {
        color: var(--color-accent);
      }
      .bottom-nav-icon {
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .bottom-nav-label {
        font-size: 10px;
        font-weight: 500;
        line-height: 1.2;
        letter-spacing: 0.01em;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .bottom-nav-link--menu {
        color: var(--color-text-secondary);
      }

      /* ── Mobile menu drawer ────────────────────────── */
      .mobile-menu-backdrop {
        position: fixed;
        inset: 0;
        background: rgb(0 0 0 / 0.5);
        z-index: 40;
        animation: fade-in 150ms ease-out;
      }
      .mobile-menu {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: min(20rem, 85vw);
        background: var(--color-surface);
        border-left: 1px solid var(--color-border);
        z-index: 50;
        display: flex;
        flex-direction: column;
        animation: slide-in-right 200ms ease-out;
        padding-bottom: env(safe-area-inset-bottom, 0);
      }
      .mobile-menu-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 16px 14px;
        border-bottom: 1px solid var(--color-border);
        min-height: 56px;
      }
      .mobile-menu-close {
        margin-left: auto;
        background: transparent;
        border: 0;
        color: var(--color-text-secondary);
        cursor: pointer;
        padding: 6px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .mobile-menu-close:hover {
        color: var(--color-text-primary);
        background: var(--color-surface-raised);
      }
      .mobile-menu-nav {
        flex: 1;
        overflow-y: auto;
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }
      .mobile-menu-link {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 12px;
        border-radius: 8px;
        text-decoration: none;
        color: var(--color-text-secondary);
        font-size: 14px;
        font-weight: 500;
        min-height: 44px;
      }
      .mobile-menu-link:hover {
        background: var(--color-surface-raised);
        color: var(--color-text-primary);
      }
      .mobile-menu-link--active {
        background: color-mix(in srgb, var(--color-accent) 12%, transparent);
        color: var(--color-accent);
      }
      .mobile-menu-icon {
        display: flex;
        align-items: center;
        flex-shrink: 0;
      }
      .mobile-menu-footer {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 14px;
        border-top: 1px solid var(--color-border);
        min-height: 48px;
      }

      @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      @keyframes slide-in-right {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
      }

      @media (prefers-reduced-motion: reduce) {
        .mobile-menu-backdrop,
        .mobile-menu {
          animation: none;
        }
      }
    `,
  ],
})
export class ShellComponent {
  private readonly auth = inject(AuthService);
  private readonly platformId = inject(PLATFORM_ID);

  readonly collapsed = signal(false);
  readonly menuOpen = signal(false);

  readonly isAuthenticated = this.auth.isAuthenticated;
  readonly isAdmin = computed(() => this.auth.role() === 'admin');

  /** Filter Admin nav item for non-admins. Server-side guard is the real enforcement. */
  readonly navItems = computed<readonly NavItem[]>(() =>
    this.isAdmin() ? NAV_ITEMS : NAV_ITEMS.filter(n => n.path !== '/admin')
  );

  /** Mobile bottom bar: max 4 items from visible nav + "More" button. */
  readonly mobileNavItems = computed<readonly NavItem[]>(() => this.navItems().slice(0, 4));

  constructor() {
    afterRenderEffect(() => {
      if (!isPlatformBrowser(this.platformId)) return;
      if (window.innerWidth < 1024) {
        this.collapsed.set(true);
      }
    });
  }

  @HostListener('window:keydown.escape')
  onEscape(): void {
    if (this.menuOpen()) this.closeMenu();
  }

  @HostListener('window:resize')
  onResize(): void {
    if (!isPlatformBrowser(this.platformId)) return;
    if (window.innerWidth >= 1024 && this.menuOpen()) {
      this.menuOpen.set(false);
    }
  }

  toggleCollapsed(): void {
    this.collapsed.update(v => !v);
  }

  toggleMenu(): void {
    this.menuOpen.update(v => !v);
  }

  closeMenu(): void {
    this.menuOpen.set(false);
  }
}
