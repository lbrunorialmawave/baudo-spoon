import {
  Component,
  Input,
  OnChanges,
  SimpleChanges,
  inject,
  signal,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { AuctionService } from '../../../core/services/auction.service';
import {
  AuctionRuleset,
  DepartmentBudgetPlan,
  DepartmentCap,
} from '../../../core/models/auction.models';

/* Same grouping used in AuctionComponent's ROLE_COLOR: classic P/D/C/A
   plus the MANTRA 12-role palette, mapped onto the classic line colors. */
const ROLE_COLOR: Record<string, string> = {
  P: 'var(--color-role-gk)',
  D: 'var(--color-role-def)',
  C: 'var(--color-role-mid)',
  A: 'var(--color-role-fwd)',
  Por: 'var(--color-role-gk)',
  Dc: 'var(--color-role-def)',
  B: 'var(--color-role-def)',
  Dd: 'var(--color-role-def)',
  Ds: 'var(--color-role-def)',
  E: 'var(--color-role-mid)',
  M: 'var(--color-role-mid)',
  T: 'var(--color-role-fwd)',
  W: 'var(--color-role-fwd)',
  Pc: 'var(--color-role-fwd)',
};

const STORAGE_KEY = 'fi.auction.deptBudgetExpanded';

@Component({
  selector: 'app-department-budget-panel',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div
      class="dept-budget-panel"
      [class.is-collapsed]="!expanded()"
      [class.is-live]="collapsible"
    >
      <button
        type="button"
        class="panel-toggle"
        (click)="toggle()"
        [attr.aria-expanded]="expanded()"
        [attr.aria-controls]="panelId"
        [id]="toggleId"
      >
        <span class="toggle-left">
          <span class="chevron" aria-hidden="true">{{ expanded() ? '▾' : '▸' }}</span>
          <span class="panel-title">Tetti di spesa per reparto</span>
          @if (plan(); as p) {
            <span class="compact-meta">{{ p.departments.length }} reparti</span>
            @if (p.warnings.length) {
              <span class="badge warn" [title]="p.warnings[0]">attenzione</span>
            }
          }
        </span>
        <span class="toggle-hint muted">
          {{ expanded() ? 'Nascondi' : 'Mostra' }}
        </span>
      </button>

      @if (expanded()) {
        <div
          class="panel-body"
          [id]="panelId"
          role="region"
          [attr.aria-labelledby]="toggleId"
        >
          <p class="panel-subtitle">
            Hard cap = limite matematico di fattibilità. Banda consigliata =
            prior di mercato + peso slot (±{{ tolerance * 100 | number: '1.0-0' }}%).
          </p>

          @if (loading()) {
            <p class="muted">Calcolo in corso…</p>
          } @else if (error()) {
            <p class="error-text">{{ error() }}</p>
          } @else if (plan(); as p) {
            @if (p.warnings.length) {
              <ul class="warnings">
                @for (w of p.warnings; track w) {
                  <li>{{ w }}</li>
                }
              </ul>
            }
            <div class="dept-list">
              @for (d of p.departments; track d.departmentId) {
                <div class="dept-row">
                  <div class="dept-meta">
                    <span class="dept-dot" [style.background]="departmentColor(d)"></span>
                    <span class="dept-label">{{ d.labelIt }}</span>
                    <span class="dept-slots">{{ d.slots }} slot</span>
                    @if (d.marketShareSource === 'fallback_slot_only') {
                      <span class="badge structural">strutturale</span>
                    }
                    @if (d.clampedToHardCap) {
                      <span class="badge clamped">clamped</span>
                    }
                  </div>
                  <div
                    class="bar-track"
                    [title]="
                      'Hard cap: ' + d.hardCap.credits + ' cr (' + d.hardCap.percent + '%)'
                    "
                  >
                    <div
                      class="bar-hard"
                      [style.width.%]="Math.min(100, d.hardCap.percent)"
                    ></div>
                    <div
                      class="bar-rec"
                      [style.left.%]="recLeft(d, p.budgetInitial)"
                      [style.width.%]="recWidth(d, p.budgetInitial)"
                      [style.background]="departmentColor(d)"
                    ></div>
                  </div>
                  <div class="dept-numbers">
                    <span class="rec">
                      consigliato {{ d.recommendedMin.credits }}–{{ d.recommendedMax.credits }}
                      cr ({{ d.recommendedMin.percent }}–{{ d.recommendedMax.percent }}%)
                    </span>
                    <span class="hard muted">
                      max assoluto {{ d.hardCap.credits }} cr ({{ d.hardCap.percent }}%)
                    </span>
                  </div>
                </div>
              }
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [
    `
      .dept-budget-panel {
        flex-shrink: 0;
        margin: 0;
        border-bottom: 1px solid var(--color-border, #27272a);
        background: var(--color-surface, #141518);
        color: var(--color-text-primary, #f4f4f5);
      }
      /* Setup context: keep card look when not in live chrome */
      .dept-budget-panel:not(.is-live) {
        margin: 12px 16px;
        border: 1px solid var(--color-border, #27272a);
        border-radius: 10px;
        border-bottom: 1px solid var(--color-border, #27272a);
      }

      .panel-toggle {
        display: flex;
        width: 100%;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 10px 16px;
        margin: 0;
        border: none;
        background: transparent;
        color: inherit;
        cursor: pointer;
        text-align: left;
        min-height: 44px;
      }
      .panel-toggle:hover {
        background: color-mix(in srgb, var(--color-text-primary, #f4f4f5) 4%, transparent);
      }
      .panel-toggle:focus-visible {
        outline: 2px solid var(--color-accent, #3b82f6);
        outline-offset: -2px;
      }
      .toggle-left {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
        flex-wrap: wrap;
      }
      .chevron {
        font-size: 11px;
        color: var(--color-text-secondary, #a1a1aa);
        width: 12px;
        flex-shrink: 0;
      }
      .panel-title {
        margin: 0;
        font-size: 13px;
        font-weight: 700;
        color: var(--color-text-primary, #f4f4f5);
      }
      .compact-meta {
        font-size: 11px;
        color: var(--color-text-secondary, #a1a1aa);
        font-variant-numeric: tabular-nums;
      }
      .toggle-hint {
        font-size: 11px;
        flex-shrink: 0;
      }

      /* Expanded body: capped height so live layout never loses the main task */
      .panel-body {
        padding: 0 16px 14px;
        max-height: min(40vh, 360px);
        overflow-y: auto;
        overscroll-behavior: contain;
        -webkit-overflow-scrolling: touch;
      }
      .dept-budget-panel:not(.is-live) .panel-body {
        max-height: none;
      }

      .panel-subtitle {
        margin: 0 0 12px;
        font-size: 11px;
        line-height: 1.45;
        color: var(--color-text-secondary, #a1a1aa);
      }
      .muted {
        font-size: 12px;
        color: var(--color-text-secondary, #a1a1aa);
      }
      .error-text {
        margin: 0;
        font-size: 12px;
        color: #ef4444;
      }
      .warnings {
        margin: 0 0 12px;
        padding: 8px 10px 8px 24px;
        border-radius: 8px;
        border: 1px solid color-mix(in srgb, #f59e0b 35%, transparent);
        background: color-mix(in srgb, #f59e0b 10%, transparent);
        font-size: 11px;
        line-height: 1.5;
        color: #f59e0b;
      }
      .dept-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .dept-row {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid var(--color-border, #27272a);
        background: var(--color-bg, #0b0c0f);
      }
      .dept-meta {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        flex-wrap: wrap;
      }
      .dept-dot {
        width: 8px;
        height: 8px;
        border-radius: 2px;
        flex-shrink: 0;
      }
      .dept-label {
        font-weight: 600;
        color: var(--color-text-primary, #f4f4f5);
      }
      .dept-slots {
        color: var(--color-text-secondary, #a1a1aa);
        font-size: 11px;
        font-variant-numeric: tabular-nums;
      }
      .badge {
        font-size: 10px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 999px;
        border: 1px solid var(--color-border, #27272a);
        color: var(--color-text-secondary, #a1a1aa);
        white-space: nowrap;
      }
      .badge.structural {
        border-color: color-mix(in srgb, var(--color-accent, #3b82f6) 45%, transparent);
        color: var(--color-accent, #3b82f6);
        background: color-mix(in srgb, var(--color-accent, #3b82f6) 12%, transparent);
      }
      .badge.clamped {
        border-color: color-mix(in srgb, #f59e0b 45%, transparent);
        color: #f59e0b;
        background: color-mix(in srgb, #f59e0b 12%, transparent);
      }
      .badge.warn {
        border-color: color-mix(in srgb, #f59e0b 45%, transparent);
        color: #f59e0b;
        background: color-mix(in srgb, #f59e0b 12%, transparent);
      }
      .bar-track {
        position: relative;
        height: 6px;
        border-radius: 999px;
        background: var(--color-border, #27272a);
        overflow: hidden;
      }
      .bar-hard {
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        border-radius: 999px;
        background: color-mix(in srgb, var(--color-text-secondary, #a1a1aa) 35%, transparent);
      }
      .bar-rec {
        position: absolute;
        top: 0;
        bottom: 0;
        border-radius: 999px;
        opacity: 0.9;
      }
      .dept-numbers {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 11px;
        font-variant-numeric: tabular-nums;
      }
      .rec {
        font-weight: 600;
        color: var(--color-text-primary, #f4f4f5);
      }
      .hard {
        color: var(--color-text-secondary, #a1a1aa);
      }
    `,
  ],
})
export class DepartmentBudgetPanelComponent implements OnChanges {
  private readonly auctionService = inject(AuctionService);
  private static nextId = 0;
  private readonly instanceId = DepartmentBudgetPanelComponent.nextId++;

  readonly panelId = `dept-budget-body-${this.instanceId}`;
  readonly toggleId = `dept-budget-toggle-${this.instanceId}`;

  @Input({ required: true }) ruleset!: AuctionRuleset | string;
  /**
   * Aligns with AuctionComponent.roleQuotas (Partial — some keys may be absent).
   * Only entries with a defined positive number are sent to the API.
   */
  @Input({ required: true }) roleQuotas!: Partial<Record<string, number>>;
  @Input({ required: true }) budgetInitial!: number;
  @Input() referenceBudget = 300;
  @Input() minSlotPrice = 1;
  @Input() alphaMarketVsSlot = 0.65;
  @Input() tolerance = 0.2;
  /**
   * When true (live tracker), panel is collapsible and starts collapsed so the
   * primary workflow (search + record) keeps the viewport. Setup view can pass
   * false to keep it always expanded.
   */
  @Input() collapsible = false;
  /** Initial expanded state when collapsible. Ignored if user has a stored preference. */
  @Input() defaultExpanded = true;

  readonly plan = signal<DepartmentBudgetPlan | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly expanded = signal(true);
  readonly Math = Math;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['collapsible'] || changes['defaultExpanded']) {
      this.initExpandedState();
    }
    this.refresh();
  }

  toggle(): void {
    const next = !this.expanded();
    this.expanded.set(next);
    if (this.collapsible) {
      try {
        sessionStorage.setItem(STORAGE_KEY, next ? '1' : '0');
      } catch {
        /* private mode / blocked storage */
      }
    }
  }

  private initExpandedState(): void {
    if (!this.collapsible) {
      this.expanded.set(true);
      return;
    }
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored === '1' || stored === '0') {
        this.expanded.set(stored === '1');
        return;
      }
    } catch {
      /* ignore */
    }
    this.expanded.set(this.defaultExpanded);
  }

  private normalizedRoleQuotas(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [role, quota] of Object.entries(this.roleQuotas ?? {})) {
      if (typeof quota === 'number' && Number.isFinite(quota) && quota > 0) {
        out[role] = quota;
      }
    }
    return out;
  }

  private refresh(): void {
    if (!this.budgetInitial || this.budgetInitial <= 0) {
      this.plan.set(null);
      return;
    }
    const quotas = this.normalizedRoleQuotas();
    if (Object.keys(quotas).length === 0) {
      this.plan.set(null);
      return;
    }
    this.loading.set(true);
    this.error.set(null);
    this.auctionService
      .getDepartmentBudgetPlan({
        ruleset: (this.ruleset as 'CLASSIC' | 'MANTRA') || 'CLASSIC',
        roleQuotas: quotas,
        budgetInitial: this.budgetInitial,
        referenceBudget: this.referenceBudget,
        minSlotPrice: this.minSlotPrice,
        alphaMarketVsSlot: this.alphaMarketVsSlot,
        tolerance: this.tolerance,
      })
      .subscribe({
        next: (p) => {
          this.plan.set(p);
          this.loading.set(false);
        },
        error: (err) => {
          this.error.set(
            err?.error?.detail || err?.message || 'Errore nel calcolo dei tetti',
          );
          this.loading.set(false);
          this.plan.set(null);
        },
      });
  }

  recLeft(d: DepartmentCap, budget: number): number {
    if (budget <= 0) return 0;
    return Math.max(0, Math.min(100, (d.recommendedMin.credits / budget) * 100));
  }

  recWidth(d: DepartmentCap, budget: number): number {
    if (budget <= 0) return 0;
    const left = (d.recommendedMin.credits / budget) * 100;
    const right = (d.recommendedMax.credits / budget) * 100;
    return Math.max(0, Math.min(100 - left, right - left));
  }

  departmentColor(d: DepartmentCap): string {
    return ROLE_COLOR[d.roles?.[0]] ?? 'var(--color-text-secondary)';
  }
}
