import { Component, Input, OnChanges, SimpleChanges, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { AuctionService } from '../../../core/services/auction.service';
import {
  AuctionRuleset,
  DepartmentBudgetPlan,
  DepartmentCap,
} from '../../../core/models/auction.models';

@Component({
  selector: 'app-department-budget-panel',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="dept-budget-panel">
      <div class="panel-header">
        <h3 class="panel-title">Tetti di spesa per reparto</h3>
        <p class="panel-subtitle">
          Hard cap = limite matematico di fattibilità.
          Banda consigliata = prior di mercato + peso slot (±{{ tolerance * 100 | number:'1.0-0' }}%).
        </p>
      </div>
      @if (loading()) {
        <p class="muted">Calcolo in corso…</p>
      } @else if (error()) {
        <p class="error">{{ error() }}</p>
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
                [title]="'Hard cap: ' + d.hardCap.credits + ' cr (' + d.hardCap.percent + '%)'"
              >
                <div
                  class="bar-hard"
                  [style.width.%]="Math.min(100, d.hardCap.percent)"
                ></div>
                <div
                  class="bar-rec"
                  [style.left.%]="recLeft(d, p.budgetInitial)"
                  [style.width.%]="recWidth(d, p.budgetInitial)"
                ></div>
              </div>
              <div class="dept-numbers">
                <span class="rec">
                  consigliato {{ d.recommendedMin.credits }}–{{ d.recommendedMax.credits }} cr
                  ({{ d.recommendedMin.percent }}–{{ d.recommendedMax.percent }}%)
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
  `,
  styles: [
    `
      .dept-budget-panel {
        margin-top: 1rem;
        padding: 0.75rem 1rem;
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 8px;
        background: var(--color-surface-2, #f8fafc);
      }
      .panel-title {
        margin: 0 0 0.25rem;
        font-size: 0.95rem;
        font-weight: 600;
      }
      .panel-subtitle {
        margin: 0 0 0.75rem;
        font-size: 0.8rem;
        color: var(--color-muted, #64748b);
      }
      .warnings {
        margin: 0 0 0.75rem;
        padding-left: 1.1rem;
        font-size: 0.8rem;
        color: var(--color-warning, #b45309);
      }
      .dept-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
      }
      .dept-row {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
      }
      .dept-meta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
      }
      .dept-label {
        font-weight: 600;
      }
      .dept-slots {
        color: var(--color-muted, #64748b);
        font-size: 0.8rem;
      }
      .badge {
        font-size: 0.7rem;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        background: #e2e8f0;
        color: #475569;
      }
      .badge.structural {
        background: #e0e7ff;
        color: #3730a3;
      }
      .badge.clamped {
        background: #fef3c7;
        color: #92400e;
      }
      .bar-track {
        position: relative;
        height: 10px;
        background: #e2e8f0;
        border-radius: 5px;
        overflow: hidden;
      }
      .bar-hard {
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        background: #cbd5e1;
        border-radius: 5px;
      }
      .bar-rec {
        position: absolute;
        top: 0;
        bottom: 0;
        background: var(--color-primary, #2563eb);
        border-radius: 5px;
        opacity: 0.85;
      }
      .dept-numbers {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        font-size: 0.78rem;
      }
      .rec {
        font-weight: 500;
      }
      .muted {
        color: var(--color-muted, #64748b);
      }
      .error {
        color: var(--color-danger, #b91c1c);
        font-size: 0.85rem;
      }
    `,
  ],
})
export class DepartmentBudgetPanelComponent implements OnChanges {
  private readonly auctionService = inject(AuctionService);

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

  readonly plan = signal<DepartmentBudgetPlan | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly Math = Math;

  ngOnChanges(_changes: SimpleChanges): void {
    this.refresh();
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
}