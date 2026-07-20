import {
  Component, computed, inject, signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { OptimizerService } from '../../core/services/optimizer.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  FormationConfig,
  MultiStrategyResult,
  OptimizationResult,
  SquadPlayer,
  StrategyProfile,
} from '../../core/models/api.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

const STRATEGY_META: Record<string, { label: string; icon: string }> = {
  BALANCED:        { label: 'Bilanciata',      icon: '⚖️' },
  SUPER_DEFENSIVE: { label: 'Super-difensiva', icon: '🛡️' },
  SUPER_OFFENSIVE: { label: 'Super-offensiva', icon: '⚡' },
  MIXED:           { label: 'Mista',           icon: '🎯' },
};

const ROLE_COLORS: Record<string, string> = {
  P: 'var(--color-role-gk)',
  D: 'var(--color-role-def)',
  C: 'var(--color-role-mid)',
  A: 'var(--color-role-fwd)',
};

const ROLE_LABELS: Record<string, string> = { P: 'GK', D: 'DEF', C: 'MID', A: 'FWD' };

const ALL_FORMATIONS: FormationConfig[] = [
  { label: '3-4-3', defenders: 3, midfielders: 4, forwards: 3 },
  { label: '4-3-3', defenders: 4, midfielders: 3, forwards: 3 },
  { label: '4-4-2', defenders: 4, midfielders: 4, forwards: 2 },
  { label: '3-5-2', defenders: 3, midfielders: 5, forwards: 2 },
];

@Component({
  selector: 'app-optimizer',
  standalone: true,
  imports: [FormsModule, DecimalPipe, SkeletonComponent, ErrorBoundaryComponent],
  template: `
    <div class="optimizer-page">

      <header class="page-header">
        <div>
          <h1 class="page-title">Squad Optimizer</h1>
          <p class="page-subtitle">ILP-based Fantacalcio squad builder · 4 strategies</p>
        </div>
      </header>

      <div class="optimizer-body">

        <!-- ── Config panel ──────────────────────────────── -->
        <aside class="config-panel card">

          <!-- BASIC -->
          <p class="section-divider">Basic</p>

          <div class="field-group">
            <label class="field-label">Season</label>
            <select class="field-input" [(ngModel)]="seasonStart">
              @for (s of seasons(); track s) {
                <option [value]="s">{{ s }}/{{ s + 1 }}</option>
              }
            </select>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Budget <span class="field-hint">cr.</span></label>
              <input class="field-input" type="number" min="200" max="1000" step="25"
                     [(ngModel)]="budget" />
            </div>
            <div class="field-group">
              <label class="field-label">Participants</label>
              <input class="field-input" type="number" min="4" max="16" step="1"
                     [(ngModel)]="numParticipants" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Min auction quota</label>
              <input class="field-input" type="number" min="0" max="10" step="1"
                     [(ngModel)]="minQtA" />
            </div>
            <div class="field-group">
              <label class="field-label">Solver timeout <span class="field-hint">s</span></label>
              <input class="field-input" type="number" min="5" max="300" step="5"
                     [(ngModel)]="solverTimeoutSeconds" />
            </div>
          </div>

          <!-- SQUAD CONSTRAINTS -->
          <p class="section-divider">Squad Constraints</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Min distinct teams</label>
              <input class="field-input" type="number" min="1" max="25" step="1"
                     [(ngModel)]="minDistinctTeams" />
            </div>
            <div class="field-group">
              <label class="field-label">Max per team</label>
              <input class="field-input" type="number" min="1" max="10" step="1"
                     [(ngModel)]="maxPlayersPerTeam" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Big-teams cap</label>
              <input class="field-input" type="number" min="0" max="25" step="1"
                     [(ngModel)]="bigTeamsCap" />
            </div>
            <div class="field-group">
              <label class="field-label">Max budget share <span class="field-hint">0–1</span></label>
              <input class="field-input" type="number" min="0.05" max="1" step="0.05"
                     [(ngModel)]="maxSinglePlayerBudgetShare" />
            </div>
          </div>

          <div class="field-group">
            <label class="field-label">Big teams <span class="field-hint">comma-separated</span></label>
            <textarea class="field-input field-textarea" rows="2"
                      [(ngModel)]="bigTeamsRaw"
                      placeholder="Inter, Milan, Juventus, Napoli"></textarea>
          </div>

          <!-- PLAYER FILTERS -->
          <p class="section-divider">Player Filters</p>

          <div class="field-group">
            <label class="field-label">Must include <span class="field-hint">player IDs, comma-separated</span></label>
            <textarea class="field-input field-textarea" rows="2"
                      [(ngModel)]="mustIncludeRaw"
                      placeholder="fm-12345, fm-67890"></textarea>
          </div>

          <div class="field-group">
            <label class="field-label">Exclude <span class="field-hint">player IDs, comma-separated</span></label>
            <textarea class="field-input field-textarea" rows="2"
                      [(ngModel)]="excludeRaw"
                      placeholder="fm-12345, fm-67890"></textarea>
          </div>

          <!-- RULESET & RISK -->
          <p class="section-divider">Ruleset & Risk</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Ruleset</label>
              <select class="field-input" [(ngModel)]="ruleset">
                <option value="CLASSIC">Classic</option>
                <option value="MANTRA">Mantra</option>
              </select>
            </div>
            <div class="field-group">
              <label class="field-label">Risk aversion <span class="field-hint">0 = neutral</span></label>
              <input class="field-input" type="number" min="0" max="5" step="0.1"
                     [(ngModel)]="riskAversion" />
            </div>
          </div>

          <!-- FORMATIONS -->
          <p class="section-divider">Formations</p>

          <div class="check-grid">
            @for (f of allFormations; track f.label) {
              <label class="check-chip" [class.active]="selectedFormations().has(f.label)">
                <input type="checkbox" [checked]="selectedFormations().has(f.label)"
                       (change)="toggleFormation(f.label)" />
                {{ f.label }}
              </label>
            }
          </div>

          <div class="field-group">
            <label class="field-label">Preferred formation <span class="field-hint">hard constraint</span></label>
            <select class="field-input" [(ngModel)]="preferredFormationLabel">
              <option value="">None</option>
              @for (f of allFormations; track f.label) {
                <option [value]="f.label">{{ f.label }}</option>
              }
            </select>
          </div>

          <!-- INFLATION MODEL -->
          <p class="section-divider">Inflation Model</p>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Percentile threshold</label>
              <input class="field-input" type="number" min="0" max="1" step="0.05"
                     [(ngModel)]="inflationPercentileThreshold" />
            </div>
            <div class="field-group">
              <label class="field-label">Max multiplier</label>
              <input class="field-input" type="number" min="1" max="5" step="0.1"
                     [(ngModel)]="maxInflationMultiplier" />
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <label class="field-label">Base rate</label>
              <input class="field-input" type="number" min="0" max="1" step="0.01"
                     [(ngModel)]="baseInflationRate" />
            </div>
            <div class="field-group">
              <label class="field-label">Baseline participants</label>
              <input class="field-input" type="number" min="2" max="20" step="1"
                     [(ngModel)]="baselineParticipants" />
            </div>
          </div>

          <!-- STRATEGIES -->
          <p class="section-divider">Strategies</p>

          <div class="check-col">
            @for (s of availableStrategies(); track s) {
              <label class="strategy-check" [class.active]="selectedStrategies().has(s)">
                <input type="checkbox" [checked]="selectedStrategies().has(s)"
                       (change)="toggleStrategy(s)" />
                <span>{{ meta(s).icon }}</span>
                <span>{{ meta(s).label }}</span>
              </label>
            }
          </div>

          <button class="run-btn" (click)="run()" [disabled]="running() || !canRun()">
            @if (running()) {
              <span class="spinner"></span> Running…
            } @else {
              Run Optimizer
            }
          </button>

          @if (error()) {
            <app-error-boundary title="Optimizer Error" [message]="error()!" />
          }
        </aside>

        <!-- ── Results panel ─────────────────────────────── -->
        <section class="results-panel">
          @if (!results()) {
            @if (running()) {
              <div class="results-placeholder">
                <div style="width:100%;max-width:480px;display:flex;flex-direction:column;gap:12px">
                  @for (_ of [1,2,3,4]; track $index) {
                    <app-skeleton height="120px" />
                  }
                </div>
              </div>
            } @else {
              <div class="results-placeholder">
                <div class="placeholder-icon">🏗️</div>
                <p class="placeholder-text">Configure and run the optimizer to see squad recommendations</p>
              </div>
            }
          } @else {
            <div class="strategy-tabs">
              @for (name of resultKeys(); track name) {
                <button class="strategy-tab"
                        [class.active]="activeStrategy() === name"
                        (click)="activeStrategy.set(name)">
                  <span>{{ meta(name).icon }}</span>
                  <span>{{ meta(name).label }}</span>
                  @if (resultFor(name); as r) {
                    <span class="tab-score">{{ r.totalProjectedScore | number:'1.1-1' }}</span>
                  }
                </button>
              }
            </div>

            @if (activeResult(); as r) {
              <div class="summary-row">
                <div class="stat-card">
                  <p class="stat-label">Projected Score</p>
                  <p class="stat-value">{{ r.totalProjectedScore | number:'1.2-2' }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Nominal Cost</p>
                  <p class="stat-value">{{ r.totalNominalCost }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Eff. Cost</p>
                  <p class="stat-value">{{ r.totalEffectiveCost | number:'1.1-1' }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Residual</p>
                  <p class="stat-value">{{ r.budgetResidual | number:'1.0-0' }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Teams</p>
                  <p class="stat-value">{{ r.distinctTeamsCount }}</p>
                </div>
                <div class="stat-card">
                  <p class="stat-label">Status</p>
                  <p class="stat-value" [class.text-green]="r.status === 'Optimal'">{{ r.status }}</p>
                </div>
              </div>

              <div class="formation-row">
                <p class="section-label">Formations</p>
                <div class="formation-chips">
                  @for (entry of formationEntries(r); track entry[0]) {
                    <span class="formation-chip" [class.ok]="entry[1]">
                      {{ entry[0] }} {{ entry[1] ? '✓' : '✗' }}
                    </span>
                  }
                </div>
              </div>

              <div class="role-breakdown">
                @for (role of ['P','D','C','A']; track role) {
                  <div class="role-strip" [style.border-color]="roleColor(role)">
                    <span class="role-label" [style.color]="roleColor(role)">{{ roleLabel(role) }}</span>
                    <span class="role-count">{{ r.roleBreakdown[role] || 0 }}</span>
                  </div>
                }
              </div>

              <div class="squad-table-wrap">
                <table class="squad-table">
                  <thead>
                    <tr>
                      <th>Role</th><th>Name</th><th class="hide-sm">Team</th>
                      <th class="num">Cost</th><th class="num hide-sm">Eff.</th><th class="num">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    @for (p of sortedSquad(r); track p.playerId) {
                      <tr>
                        <td>
                          <span class="role-badge" [style.color]="roleColor(p.role)"
                                [style.border-color]="roleColor(p.role)">
                            {{ roleLabel(p.role) }}
                          </span>
                        </td>
                        <td class="player-name">{{ p.name }}</td>
                        <td class="team-name hide-sm">{{ p.realTeam }}</td>
                        <td class="num">{{ p.cost }}</td>
                        <td class="num faded hide-sm">{{ p.effectiveCost | number:'1.1-1' }}</td>
                        <td class="num accent">{{ p.projectedScore | number:'1.2-2' }}</td>
                      </tr>
                    }
                  </tbody>
                </table>
              </div>
            }
          }
        </section>

      </div>
    </div>
  `,
  styles: [`
    .optimizer-page { display:flex; flex-direction:column; height:100%; overflow:hidden; }
    .page-header { padding:16px; border-bottom:1px solid var(--color-border); flex-shrink:0; }
    @media (min-width: 640px) { .page-header { padding:20px 24px 16px; } }
    .page-title { font-size:16px; font-weight:700; color:var(--color-text-primary); margin:0; }
    @media (min-width: 640px) { .page-title { font-size:18px; } }
    .page-subtitle { font-size:11px; color:var(--color-text-secondary); margin:2px 0 0; }
    @media (min-width: 640px) { .page-subtitle { font-size:12px; } }

    .optimizer-body {
      display:flex; flex-direction:column;
      flex:1; overflow:hidden; min-height:0;
    }
    @media (min-width: 768px) {
      .optimizer-body {
        display:grid; grid-template-columns:300px 1fr;
      }
    }

    /* Config panel */
    .config-panel {
      border-radius:0; border-top:none; border-bottom:1px solid var(--color-border);
      border-left:none; border-right:none;
      padding:16px; overflow-y:auto; max-height:50vh;
      display:flex; flex-direction:column; gap:10px;
    }
    @media (min-width: 768px) {
      .config-panel {
        max-height:none; border-bottom:none;
        border-right:1px solid var(--color-border);
      }
    }
    .section-divider {
      font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:0.08em; color:var(--color-text-secondary);
      margin:6px 0 0; padding-bottom:6px;
      border-bottom:1px solid var(--color-border);
    }
    .field-group { display:flex; flex-direction:column; gap:4px; min-width:0; }
    .field-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .field-label { font-size:11px; font-weight:500; color:var(--color-text-secondary); }
    .field-hint { font-size:10px; opacity:0.6; }
    .field-input {
      background:var(--color-bg); border:1px solid var(--color-border);
      border-radius:6px; padding:8px;
      color:var(--color-text-primary); font-size:13px;
      outline:none; width:100%; min-width:0;
    }
    @media (min-width: 640px) {
      .field-input { padding:6px 8px; font-size:12px; }
    }
    .field-input:focus { border-color:var(--color-accent); }
    .field-textarea { resize:vertical; font-family:var(--font-sans); }

    /* Formations check grid */
    .check-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .check-chip {
      display:flex; align-items:center; justify-content:center;
      padding:8px; border-radius:6px;
      border:1px solid var(--color-border); background:var(--color-bg);
      cursor:pointer; font-size:12px; font-weight:500;
      color:var(--color-text-secondary);
      transition:border-color 100ms, color 100ms;
      min-height:36px;
    }
    .check-chip.active {
      border-color:var(--color-accent); color:var(--color-text-primary);
      background:color-mix(in srgb, var(--color-accent) 8%, transparent);
    }
    .check-chip input { display:none; }

    /* Strategies column */
    .check-col { display:flex; flex-direction:column; gap:5px; }
    .strategy-check {
      display:flex; align-items:center; gap:8px;
      padding:8px 10px; border-radius:6px;
      border:1px solid var(--color-border); background:var(--color-bg);
      cursor:pointer; font-size:12px; color:var(--color-text-secondary);
      transition:border-color 100ms, color 100ms;
      min-height:36px;
    }
    .strategy-check.active {
      border-color:var(--color-accent); color:var(--color-text-primary);
      background:color-mix(in srgb, var(--color-accent) 8%, transparent);
    }
    .strategy-check input { display:none; }

    .run-btn {
      margin-top:4px; width:100%; padding:10px;
      border-radius:8px; background:var(--color-accent);
      color:#fff; font-size:13px; font-weight:600;
      border:none; cursor:pointer;
      display:flex; align-items:center; justify-content:center; gap:6px;
      transition:opacity 120ms;
      min-height:40px;
    }
    .run-btn:disabled { opacity:0.5; cursor:not-allowed; }
    .run-btn:not(:disabled):hover { opacity:0.9; }
    .spinner {
      width:14px; height:14px;
      border:2px solid rgba(255,255,255,0.3);
      border-top-color:#fff; border-radius:50%;
      animation:spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform:rotate(360deg); } }

    /* Results panel */
    .results-panel {
      display:flex; flex-direction:column; overflow:hidden;
      min-height:0;
    }
    @media (min-width: 768px) {
      .results-panel { border-left:none; }
    }
    .results-placeholder {
      flex:1; display:flex; flex-direction:column;
      align-items:center; justify-content:center;
      gap:12px; padding:24px 16px; overflow-y:auto;
    }
    @media (min-width: 640px) { .results-placeholder { padding:32px; } }
    .placeholder-icon { font-size:40px; }
    .placeholder-text {
      font-size:13px; color:var(--color-text-secondary);
      text-align:center; max-width:280px;
    }

    .strategy-tabs {
      display:flex; border-bottom:1px solid var(--color-border);
      flex-shrink:0; padding:0 12px; overflow-x:auto;
      -webkit-overflow-scrolling:touch;
    }
    @media (min-width: 640px) { .strategy-tabs { padding:0 16px; } }
    .strategy-tab {
      display:flex; align-items:center; gap:6px;
      padding:12px 10px; border:none; background:none;
      color:var(--color-text-secondary); font-size:12px; font-weight:500;
      border-bottom:2px solid transparent; cursor:pointer;
      white-space:nowrap; transition:color 100ms, border-color 100ms;
      min-height:44px;
    }
    @media (min-width: 640px) { .strategy-tab { padding:12px 14px; } }
    .strategy-tab.active { color:var(--color-accent); border-bottom-color:var(--color-accent); }
    .tab-score {
      background:var(--color-surface-raised); border-radius:9999px;
      padding:1px 7px; font-size:11px; color:var(--color-text-secondary);
    }

    .summary-row {
      display:grid; grid-template-columns:repeat(2,1fr);
      border-bottom:1px solid var(--color-border); flex-shrink:0;
    }
    @media (min-width: 640px) { .summary-row { grid-template-columns:repeat(3,1fr); } }
    @media (min-width: 1024px) { .summary-row { grid-template-columns:repeat(6,1fr); } }
    .stat-card {
      padding:10px 12px;
      border-right:1px solid var(--color-border);
      border-bottom:1px solid var(--color-border);
    }
    @media (min-width: 640px) {
      .stat-card:nth-child(2n) { border-right:none; }
      .stat-card:nth-last-child(-n+2) { border-bottom:none; }
    }
    @media (min-width: 1024px) {
      .stat-card { padding:12px 14px; border-bottom:none; }
      .stat-card { border-right:1px solid var(--color-border); }
      .stat-card:last-child { border-right:none; }
    }
    .stat-label {
      font-size:10px; font-weight:500; text-transform:uppercase;
      letter-spacing:0.06em; color:var(--color-text-secondary); margin:0 0 3px;
    }
    .stat-value {
      font-size:14px; font-weight:700;
      font-variant-numeric:tabular-nums; color:var(--color-text-primary); margin:0;
    }
    @media (min-width: 1024px) { .stat-value { font-size:16px; } }
    .text-green { color:#22C55E !important; }

    .formation-row {
      display:flex; align-items:center; gap:10px; flex-wrap:wrap;
      padding:8px 12px; border-bottom:1px solid var(--color-border); flex-shrink:0;
    }
    @media (min-width: 640px) { .formation-row { padding:8px 16px; } }
    .section-label {
      font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:0.06em; color:var(--color-text-secondary); margin:0; white-space:nowrap;
    }
    .formation-chips { display:flex; flex-wrap:wrap; gap:4px; }
    .formation-chip {
      padding:2px 8px; border-radius:9999px; font-size:11px; font-weight:500;
      background:var(--color-surface-raised); color:var(--color-text-secondary);
      border:1px solid var(--color-border);
    }
    .formation-chip.ok {
      background:color-mix(in srgb,#22C55E 12%,transparent);
      color:#22C55E; border-color:#22C55E;
    }

    .role-breakdown {
      display:grid; grid-template-columns:repeat(2,1fr);
      flex-shrink:0; border-bottom:1px solid var(--color-border);
    }
    @media (min-width: 640px) { .role-breakdown { grid-template-columns:repeat(4,1fr); } }
    .role-strip {
      display:flex; align-items:center; justify-content:space-between;
      padding:6px 12px;
      border-right:1px solid var(--color-border);
      border-bottom:1px solid var(--color-border);
      border-left:3px solid transparent;
    }
    @media (min-width: 640px) {
      .role-strip { border-bottom:none; padding:6px 14px; }
    }
    .role-strip:nth-child(2n) { border-right:none; }
    @media (min-width: 640px) { .role-strip:nth-child(2n) { border-right:1px solid var(--color-border); } }
    .role-strip:last-child { border-right:none; }
    .role-label { font-size:11px; font-weight:600; }
    .role-count { font-size:14px; font-weight:700; color:var(--color-text-primary); }

    .squad-table-wrap {
      flex:1; overflow:auto; min-height:0;
      margin:0 -12px;
    }
    @media (min-width: 640px) { .squad-table-wrap { margin:0 -16px; } }
    .squad-table { width:100%; min-width:520px; border-collapse:collapse; font-size:13px; }
    .hide-sm { display:none; }
    @media (min-width: 640px) { .hide-sm { display:table-cell; } }
    .squad-table thead th {
      position:sticky; top:0; z-index:1;
      background:var(--color-surface); padding:8px 10px; text-align:left;
      font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:0.05em; color:var(--color-text-secondary);
      border-bottom:1px solid var(--color-border);
    }
    @media (min-width: 640px) { .squad-table thead th { padding:8px 14px; } }
    .squad-table tbody tr {
      border-bottom:1px solid var(--color-border); transition:background 100ms;
    }
    .squad-table tbody tr:hover { background:var(--color-surface-raised); }
    .squad-table tbody td { padding:8px 10px; color:var(--color-text-primary); }
    @media (min-width: 640px) { .squad-table tbody td { padding:8px 14px; } }
    .squad-table .num { text-align:right; font-variant-numeric:tabular-nums; }
    .role-badge {
      display:inline-flex; align-items:center; justify-content:center;
      width:36px; padding:1px 0; border-radius:4px;
      border:1px solid; font-size:10px; font-weight:700;
    }
    .player-name { font-weight:500; }
    .team-name { color:var(--color-text-secondary); font-size:12px; }
    .faded { color:var(--color-text-secondary); }
    .accent { color:var(--color-accent); font-weight:600; }
  `],
})
export class OptimizerComponent {
  private readonly optimizerService = inject(OptimizerService);
  private readonly quotationService = inject(QuotationService);

  readonly allFormations = ALL_FORMATIONS;

  // Strategies loaded from API; fallback to known names if unavailable
  readonly availableStrategies = signal<string[]>(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']);

  // ── Basic ─────────────────────────────────────────────
  readonly seasons = signal<number[]>([]);
  readonly seasonStart = signal<number>(2024);
  readonly budget = signal(500);
  readonly numParticipants = signal(8);
  readonly minQtA = signal(1);
  readonly solverTimeoutSeconds = signal(30);

  // ── Squad constraints ─────────────────────────────────
  readonly minDistinctTeams = signal(12);
  readonly maxPlayersPerTeam = signal(4);
  readonly bigTeamsCap = signal(10);
  readonly bigTeamsRaw = signal('Inter, Milan, Juventus, Napoli');
  readonly maxSinglePlayerBudgetShare = signal(0.30);

  // ── Must include / exclude ─────────────────────────────
  readonly mustIncludeRaw = signal('');
  readonly excludeRaw = signal('');

  // ── Formations ────────────────────────────────────────
  readonly selectedFormations = signal(new Set(ALL_FORMATIONS.map(f => f.label)));
  readonly preferredFormationLabel = signal<string>('');

  // ── Ruleset ───────────────────────────────────────────
  readonly ruleset = signal<'CLASSIC' | 'MANTRA'>('CLASSIC');

  // ── Inflation model ───────────────────────────────────
  readonly inflationPercentileThreshold = signal(0.7);
  readonly maxInflationMultiplier = signal(1.6);
  readonly baseInflationRate = signal(0.05);
  readonly baselineParticipants = signal(8);

  // ── Risk ──────────────────────────────────────────────
  readonly riskAversion = signal(0.0);

  // ── Strategies ────────────────────────────────────────
  readonly selectedStrategies = signal(new Set(['BALANCED', 'SUPER_DEFENSIVE', 'SUPER_OFFENSIVE', 'MIXED']));

  // ── Results ───────────────────────────────────────────
  readonly running = signal(false);
  readonly error = signal<string | null>(null);
  readonly results = signal<MultiStrategyResult | null>(null);
  readonly activeStrategy = signal<string>('');

  readonly resultKeys = computed(() => Object.keys(this.results()?.results ?? {}));
  readonly activeResult = computed((): OptimizationResult | null =>
    this.results()?.results[this.activeStrategy()] ?? null,
  );
  readonly canRun = computed(() =>
    this.selectedStrategies().size > 0 &&
    this.selectedFormations().size > 0 &&
    this.seasons().length > 0,
  );

  constructor() {
    this.quotationService.getSeasons().subscribe({
      next: s => {
        const sorted = [...s].sort((a, b) => b - a);
        this.seasons.set(sorted);
        if (sorted.length) this.seasonStart.set(sorted[0]);
      },
      error: () => { this.seasons.set([2024, 2023, 2022]); },
    });

    this.optimizerService.getStrategies().subscribe({
      next: res => {
        const names = res.strategies.map((s: StrategyProfile) => s.name);
        this.availableStrategies.set(names);
        this.selectedStrategies.set(new Set(names));
      },
      error: () => { /* keep fallback */ },
    });
  }

  toggleStrategy(name: string): void {
    this.selectedStrategies.update(s => {
      const n = new Set(s); n.has(name) ? n.delete(name) : n.add(name); return n;
    });
  }

  toggleFormation(label: string): void {
    this.selectedFormations.update(s => {
      const n = new Set(s); n.has(label) ? n.delete(label) : n.add(label); return n;
    });
  }

  run(): void {
    this.running.set(true);
    this.error.set(null);

    const bigTeams = this.bigTeamsRaw()
      .split(',').map(t => t.trim()).filter(Boolean);

    const formations = ALL_FORMATIONS.filter(f => this.selectedFormations().has(f.label));

    const mustInclude = this.mustIncludeRaw()
      .split(',').map(s => s.trim()).filter(Boolean);

    const exclude = this.excludeRaw()
      .split(',').map(s => s.trim()).filter(Boolean);

    const preferredLabel = this.preferredFormationLabel();
    const preferredFormation = preferredLabel
      ? (ALL_FORMATIONS.find(f => f.label === preferredLabel) ?? null)
      : null;

    this.optimizerService.runMulti({
      seasonStart: this.seasonStart(),
      budget: this.budget(),
      numParticipants: this.numParticipants(),
      minQtA: this.minQtA(),
      solverTimeoutSeconds: this.solverTimeoutSeconds(),
      minDistinctTeams: this.minDistinctTeams(),
      maxPlayersPerTeam: this.maxPlayersPerTeam(),
      bigTeamsCap: this.bigTeamsCap(),
      bigTeams,
      formations,
      inflationConfig: {
        inflationPercentileThreshold: this.inflationPercentileThreshold(),
        maxInflationMultiplier: this.maxInflationMultiplier(),
        baseInflationRate: this.baseInflationRate(),
        baselineParticipants: this.baselineParticipants(),
      },
      maxSinglePlayerBudgetShare: this.maxSinglePlayerBudgetShare(),
      mustInclude: mustInclude.length ? mustInclude : undefined,
      exclude: exclude.length ? exclude : undefined,
      ruleset: this.ruleset(),
      preferredFormation,
      riskAversion: this.riskAversion(),
      strategyNames: [...this.selectedStrategies()],
    }).subscribe({
      next: res => {
        this.results.set(res);
        this.activeStrategy.set(Object.keys(res.results)[0] ?? '');
        this.running.set(false);
      },
      error: err => {
        this.error.set(err.error?.detail ?? err.message ?? 'Unknown error');
        this.running.set(false);
      },
    });
  }

  resultFor(name: string): OptimizationResult | null {
    return this.results()?.results[name] ?? null;
  }

  sortedSquad(r: OptimizationResult): SquadPlayer[] {
    const order = ['P', 'D', 'C', 'A'];
    return [...r.squad].sort((a, b) =>
      order.indexOf(a.role) - order.indexOf(b.role) || b.projectedScore - a.projectedScore,
    );
  }

  formationEntries(r: OptimizationResult): [string, boolean][] {
    return Object.entries(r.formationFeasibility);
  }

  meta(name: string) {
    return STRATEGY_META[name] ?? { label: name, icon: '📋' };
  }

  roleColor(role: string): string { return ROLE_COLORS[role] ?? 'var(--color-text-secondary)'; }
  roleLabel(role: string): string { return ROLE_LABELS[role] ?? role; }
}
