import {
  Component, computed, inject, signal, DestroyRef,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { forkJoin, Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { AuctionService } from '../../core/services/auction.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  AUCTION_ROLES,
  AuctionParticipantSetup,
  AuctionParticipantState,
  AuctionPlayerSummary,
  AuctionRole,
  AuctionSummary,
  AuctionTier,
  ProjectionResponse,
  AlternativesResponse,
} from '../../core/models/auction.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

const ROLE_COLOR: Record<string, string> = {
  P: 'var(--color-role-gk)',
  D: 'var(--color-role-def)',
  C: 'var(--color-role-mid)',
  A: 'var(--color-role-fwd)',
};

const TIER_COLOR: Record<AuctionTier, string> = {
  LOW: 'var(--color-text-secondary)',
  MID: 'var(--color-accent)',
  TOP: '#F59E0B',
};

function makeParticipants(
  n: number, budget: number, existing: AuctionParticipantSetup[] = [],
): AuctionParticipantSetup[] {
  return Array.from({ length: n }, (_, i) => existing[i] ?? {
    participantId: `team_${i + 1}`,
    displayName: `Team ${i + 1}`,
    budgetInitial: budget,
  });
}

@Component({
  selector: 'app-auction',
  standalone: true,
  imports: [FormsModule, DecimalPipe, SkeletonComponent, ErrorBoundaryComponent],
  template: `
    @if (sessionId()) {

      <!-- ═══════════════════════ LIVE VIEW ═══════════════════════ -->
      <div class="auction-page">

        <header class="page-header">
          <div>
            <h1 class="page-title">Auction Tracker</h1>
            <p class="page-subtitle">
              Session: <code class="session-id">{{ sessionId()!.slice(0, 12) }}…</code>
            </p>
          </div>
          <div class="header-actions">
            <button class="secondary-btn" (click)="saveToFile()">Save Session</button>
            <button class="danger-btn" (click)="endSession()">End Session</button>
          </div>
        </header>

        <!-- Price index strip -->
        @if (summary(); as s) {
          <div class="price-strip">
            @for (role of allRoles; track role) {
              <div class="price-role-group">
                <span class="price-role-label" [style.color]="roleColor(role)">{{ role }}</span>
                @for (tier of allTiers; track tier) {
                  @if (s.priceIndex[role]?.[tier] !== undefined) {
                    <span class="price-chip" [style.border-color]="tierColor(tier)"
                          [style.color]="tierColor(tier)">
                      {{ tier.charAt(0) }}&thinsp;{{ s.priceIndex[role]![tier]! | number:'1.2-2' }}
                    </span>
                  }
                }
              </div>
            }
          </div>
        }

        <div class="auction-body">

          <!-- ── Left: Participants ──────────────────────── -->
          <aside class="participants-panel">
            <p class="panel-heading">Standings</p>

            @if (summaryLoading() && !summary()) {
              @for (_ of [1,2,3,4,5,6,7,8]; track $index) {
                <app-skeleton height="72px" />
              }
            }

            @if (summary(); as s) {
              @for (p of s.participants; track p.participantId) {
                <div class="participant-card">
                  <div class="participant-header">
                    <span class="participant-name">{{ p.displayName }}</span>
                    <span class="participant-budget" [style.color]="budgetColor(p)">
                      {{ p.budgetResidual }} cr.
                    </span>
                  </div>
                  <div class="budget-bar">
                    <div class="budget-bar-fill"
                         [style.width]="budgetPercent(p) + '%'"
                         [style.background]="budgetColor(p)"></div>
                  </div>
                  <div class="role-chips">
                    @for (role of allRoles; track role) {
                      @if (p.roleBreakdown[role]) {
                        <span class="role-chip"
                              [style.color]="roleColor(role)"
                              [style.border-color]="roleColor(role)">
                          {{ role }}&thinsp;{{ p.roleBreakdown[role] }}
                        </span>
                      }
                    }
                    @if (p.squad.length === 0) {
                      <span class="empty-squad">—</span>
                    }
                  </div>
                </div>
              }
            }
          </aside>

          <!-- ── Right: Main area ──────────────────────── -->
          <main class="auction-main">

            <div class="action-row">

              <!-- Lookup card -->
              <div class="card">
                <p class="card-section-label">Player Lookup</p>
                <div class="pool-autocomplete">
                  <div class="lookup-row">
                    <input class="field-input" placeholder="Cerca giocatore…"
                           [ngModel]="lookupQuery"
                           (ngModelChange)="lookupQuery = $event; onPoolQueryChange($event)"
                           (keydown.escape)="poolOpen.set(false)"
                           (keydown.enter)="lookupPlayer()"
                           autocomplete="off" />
                    @if (lookupLoading()) {
                      <span class="spinner-sm" style="flex-shrink:0;color:var(--color-accent)"></span>
                    }
                  </div>
                  @if (poolOpen() && poolSuggestions().length) {
                    <ul class="pool-dropdown" role="listbox">
                      @for (p of poolSuggestions(); track p.playerId) {
                        <li class="pool-option" role="option"
                            (mousedown)="selectPoolPlayer(p)">
                          <span class="pool-name">{{ p.name }}</span>
                          <span class="pool-meta">
                            <span class="role-badge"
                                  [style.color]="roleColor(p.role)"
                                  [style.border-color]="roleColor(p.role)">{{ p.role }}</span>
                            {{ p.realTeam }} · {{ p.cost }} cr.
                          </span>
                        </li>
                      }
                    </ul>
                  }
                </div>

                @if (lookupError()) {
                  <p class="inline-error">{{ lookupError() }}</p>
                }

                @if (projection(); as proj) {
                  <div class="projection-row">
                    <span class="proj-label">Expected price</span>
                    <span class="proj-price">{{ proj.expectedPrice | number:'1.0-0' }} cr.</span>
                    <span class="tier-badge" [style.color]="tierColor(proj.tier)"
                          [style.border-color]="tierColor(proj.tier)">{{ proj.tier }}</span>
                  </div>
                }

                @if (altResult(); as alt) {
                  <div class="alternatives-grid">
                    @if (alt.lowCostAlternative; as lc) {
                      <div class="alt-card">
                        <p class="alt-label">Low Cost</p>
                        <p class="alt-name">{{ lc.name }}</p>
                        <p class="alt-meta">{{ lc.realTeam }} · {{ lc.role }} · {{ lc.cost }} cr.</p>
                      </div>
                    }
                    @if (alt.closestAlternative; as cl) {
                      <div class="alt-card">
                        <p class="alt-label">Closest</p>
                        <p class="alt-name">{{ cl.name }}</p>
                        <p class="alt-meta">{{ cl.realTeam }} · {{ cl.role }} · {{ cl.cost }} cr.</p>
                      </div>
                    }
                    @if (!alt.lowCostAlternative && !alt.closestAlternative && alt.reasonIfNone) {
                      <p class="alt-none">{{ alt.reasonIfNone }}</p>
                    }
                  </div>
                }
              </div>

              <!-- Record card -->
              <div class="card">
                <p class="card-section-label">Record Assignment</p>

                <div class="field-group">
                  <label class="field-label">Giocatore</label>
                  <input class="field-input" [ngModel]="recordPlayerName || recordPlayerId"
                         readonly placeholder="seleziona dal Lookup →"
                         [style.color]="recordPlayerId ? 'var(--color-text-primary)' : 'var(--color-text-secondary)'" />
                </div>

                <div class="field-group">
                  <label class="field-label">Winner</label>
                  <select class="field-input" [(ngModel)]="recordWinnerId">
                    <option value="">— select —</option>
                    @if (summary(); as s) {
                      @for (p of s.participants; track p.participantId) {
                        <option [value]="p.participantId">{{ p.displayName }}</option>
                      }
                    }
                  </select>
                </div>

                <div class="field-group">
                  <label class="field-label">Final Price <span class="field-hint">cr.</span></label>
                  <input class="field-input" type="number" min="1" [(ngModel)]="recordPrice" />
                </div>

                @if (recordError()) {
                  <div class="inline-rejection">
                    @if (recordRejectionCode()) {
                      <code class="rejection-code">{{ recordRejectionCode() }}</code>
                    }
                    <p class="rejection-msg">{{ recordError() }}</p>
                  </div>
                }

                <div class="record-actions">
                  <button class="run-btn"
                          (click)="submitRecord()"
                          [disabled]="recordLoading() || !recordPlayerId || !recordWinnerId || recordPrice < 1">
                    @if (recordLoading()) {
                      <span class="spinner"></span> Recording…
                    } @else {
                      Record
                    }
                  </button>
                  <button class="secondary-btn" (click)="undoLast()" [disabled]="undoLoading()">
                    @if (undoLoading()) { <span class="spinner-sm"></span> } @else { Undo }
                  </button>
                </div>
              </div>

            </div>

            <!-- Assignment history -->
            @if (reversedAssignments().length) {
              <div class="card history-card">
                <p class="card-section-label">History ({{ reversedAssignments().length }})</p>
                <div class="history-table-wrap">
                  <table class="squad-table">
                    <thead>
                      <tr>
                        <th>#</th><th>Player</th><th>Winner</th>
                        <th>R</th><th class="num">Price</th>
                        <th>Tier</th><th class="num">Δ Index</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (a of reversedAssignments(); track a.sequenceNumber) {
                        <tr>
                          <td class="seq">{{ a.sequenceNumber }}</td>
                          <td>
                            <p class="player-name">{{ a.player.name }}</p>
                            <p class="team-name">{{ a.player.realTeam }}</p>
                          </td>
                          <td>{{ winnerName(a.winnerParticipantId) }}</td>
                          <td>
                            <span class="role-badge"
                                  [style.color]="roleColor(a.role)"
                                  [style.border-color]="roleColor(a.role)">{{ a.role }}</span>
                          </td>
                          <td class="num accent">{{ a.finalPrice }}</td>
                          <td>
                            <span class="tier-badge"
                                  [style.color]="tierColor(a.tier)"
                                  [style.border-color]="tierColor(a.tier)">{{ a.tier }}</span>
                          </td>
                          <td class="num faded">
                            {{ (a.priceIndexAfter - a.priceIndexBefore) | number:'+1.3-3' }}
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            } @else if (summary()) {
              <div class="card empty-history">
                <p>No assignments yet. Look up a player, then record their sale.</p>
              </div>
            }

          </main>
        </div>
      </div>

    } @else {

      <!-- ═══════════════════════ SETUP VIEW ══════════════════════ -->
      <div class="auction-page">

        <header class="page-header">
          <div>
            <h1 class="page-title">Auction Tracker</h1>
            <p class="page-subtitle">ILP-based market drift · EWMA price index</p>
          </div>
        </header>

        <div class="setup-body">

          <!-- Config panel -->
          <aside class="config-panel card">

            <p class="section-divider">Session</p>

            <div class="field-group">
              <label class="field-label">Season</label>
              <select class="field-input" [(ngModel)]="seasonStart">
                @for (s of seasons(); track s) {
                  <option [value]="s">{{ s }}/{{ s + 1 }}</option>
                }
              </select>
            </div>

            <p class="section-divider">Participants</p>

            <div class="field-row">
              <div class="field-group">
                <label class="field-label">Count</label>
                <input class="field-input" type="number" min="2" max="20"
                       [(ngModel)]="numParticipants"
                       (change)="resizeParticipants()" />
              </div>
              <div class="field-group">
                <label class="field-label">Budget each</label>
                <input class="field-input" type="number" min="100" max="2000" step="25"
                       [ngModel]="defaultBudget"
                       (ngModelChange)="defaultBudget = +$event; applyDefaultBudget()" />
              </div>
            </div>

            <p class="section-divider">Role Quotas</p>

            <div class="quota-grid">
              @for (role of allRoles; track role) {
                <div class="field-group">
                  <label class="field-label" [style.color]="roleColor(role)">{{ role }}</label>
                  <input class="field-input" type="number" min="1" max="20"
                         [ngModel]="roleQuotas[role] ?? 0"
                         (ngModelChange)="roleQuotas[role] = +$event" />
                </div>
              }
            </div>

            <p class="section-divider">Inflation</p>

            <label class="strategy-check" [class.active]="useInflationBaseline">
              <input type="checkbox" [(ngModel)]="useInflationBaseline" />
              <span>Use inflation baseline</span>
            </label>

            <div class="field-row">
              <div class="field-group">
                <label class="field-label">Reference budget <span class="field-hint">cr.</span></label>
                <input class="field-input" type="number" min="1" max="10000" step="50"
                       [(ngModel)]="referenceBudget" />
              </div>
              <div class="field-group">
                <label class="field-label">Session budget <span class="field-hint">cr.</span></label>
                <input class="field-input" type="number" min="1" max="10000" step="50"
                       [(ngModel)]="budgetInitial" />
              </div>
            </div>

            <!-- Advanced -->
            <button class="advanced-toggle" (click)="showAdvanced = !showAdvanced">
              Advanced {{ showAdvanced ? '▲' : '▼' }}
            </button>

            @if (showAdvanced) {
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label">EWMA alpha</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="alpha" />
                </div>
                <div class="field-group">
                  <label class="field-label">Spillover adj.</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.01"
                         [(ngModel)]="spilloverAdj" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label">Spillover cross</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.01"
                         [(ngModel)]="spilloverCross" />
                </div>
                <div class="field-group">
                  <label class="field-label">Low-cost pct.</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="lowCostPercentile" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label">Min index</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.1"
                         [(ngModel)]="minIndex" />
                </div>
                <div class="field-group">
                  <label class="field-label">Max index</label>
                  <input class="field-input" type="number" min="1" max="5" step="0.1"
                         [(ngModel)]="maxIndex" />
                </div>
              </div>
              <div class="field-row">
                <div class="field-group">
                  <label class="field-label">Tier low</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="tierLow" />
                </div>
                <div class="field-group">
                  <label class="field-label">Tier top</label>
                  <input class="field-input" type="number" min="0" max="1" step="0.05"
                         [(ngModel)]="tierTop" />
                </div>
              </div>
            }

            @if (initError()) {
              <app-error-boundary title="Session Error" [message]="initError()!" />
            }

            <button class="run-btn" (click)="startAuction()" [disabled]="starting() || !seasons().length">
              @if (starting()) {
                <span class="spinner"></span> Starting…
              } @else {
                Start Auction
              }
            </button>

            <button class="secondary-btn full-w" (click)="fileInput.click()" [disabled]="starting()">
              Resume from Save
            </button>
            <input #fileInput type="file" accept=".json" style="display:none"
                   (change)="onResumeFile($event)" />

          </aside>

          <!-- Participants editor -->
          <section class="setup-right">
            <div class="card">
              <p class="card-section-label" style="margin-bottom:12px">
                Participants ({{ participants().length }})
              </p>
              <div class="participants-list">
                <div class="participants-list-header">
                  <span>Name</span><span>Budget</span>
                </div>
                @for (p of participants(); track p.participantId; let i = $index) {
                  <div class="participant-edit-row">
                    <input class="field-input"
                           [ngModel]="p.displayName"
                           (ngModelChange)="updateName(i, $event)"
                           [placeholder]="'Team ' + (i + 1)" />
                    <input class="field-input budget-input" type="number"
                           [ngModel]="p.budgetInitial"
                           (ngModelChange)="updateBudget(i, +$event)"
                           min="100" max="2000" step="25" />
                  </div>
                }
              </div>
            </div>
          </section>

        </div>
      </div>
    }
  `,
  styles: [`
    /* ── Shared layout ───────────────────────────────────────── */
    .auction-page { display:flex; flex-direction:column; height:100%; overflow:hidden; }
    .page-header {
      padding:20px 24px 16px; border-bottom:1px solid var(--color-border);
      flex-shrink:0; display:flex; align-items:flex-start; justify-content:space-between;
    }
    .page-title { font-size:18px; font-weight:700; color:var(--color-text-primary); margin:0; }
    .page-subtitle { font-size:12px; color:var(--color-text-secondary); margin:2px 0 0; }
    .session-id { font-family:var(--font-mono); font-size:11px; }
    .header-actions { display:flex; gap:8px; align-items:center; }

    /* ── Buttons ─────────────────────────────────────────────── */
    .run-btn {
      width:100%; padding:10px; border-radius:8px;
      background:var(--color-accent); color:#fff;
      font-size:13px; font-weight:600; border:none; cursor:pointer;
      display:flex; align-items:center; justify-content:center; gap:6px;
      transition:opacity 120ms; flex-shrink:0;
    }
    .run-btn:disabled { opacity:0.5; cursor:not-allowed; }
    .run-btn:not(:disabled):hover { opacity:0.9; }

    .secondary-btn {
      padding:8px 14px; border-radius:8px;
      border:1px solid var(--color-border); background:var(--color-bg);
      color:var(--color-text-secondary); font-size:12px; font-weight:500;
      cursor:pointer; transition:border-color 120ms, color 120ms; white-space:nowrap;
    }
    .secondary-btn:hover { border-color:var(--color-accent); color:var(--color-text-primary); }
    .secondary-btn:disabled { opacity:0.5; cursor:not-allowed; }
    .secondary-btn.full-w { width:100%; }

    .danger-btn {
      padding:8px 14px; border-radius:8px;
      border:1px solid #EF4444; background:transparent;
      color:#EF4444; font-size:12px; font-weight:500;
      cursor:pointer; transition:background 120ms; white-space:nowrap;
    }
    .danger-btn:hover { background:color-mix(in srgb,#EF4444 12%,transparent); }

    .icon-btn {
      padding:6px 12px; border-radius:6px;
      background:var(--color-accent); color:#fff;
      font-size:12px; font-weight:600; border:none;
      cursor:pointer; white-space:nowrap; flex-shrink:0;
      display:flex; align-items:center; gap:4px;
      transition:opacity 120ms;
    }
    .icon-btn:disabled { opacity:0.5; cursor:not-allowed; }

    /* ── Spinners ────────────────────────────────────────────── */
    .spinner {
      width:14px; height:14px; border:2px solid rgba(255,255,255,0.3);
      border-top-color:#fff; border-radius:50%; animation:spin 0.7s linear infinite;
    }
    .spinner-sm {
      width:12px; height:12px; border:2px solid rgba(255,255,255,0.3);
      border-top-color:currentColor; border-radius:50%; animation:spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform:rotate(360deg); } }

    /* ── Form fields ─────────────────────────────────────────── */
    .field-group { display:flex; flex-direction:column; gap:4px; }
    .field-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .field-label { font-size:11px; font-weight:500; color:var(--color-text-secondary); }
    .field-hint { font-size:10px; opacity:0.6; }
    .field-input {
      background:var(--color-bg); border:1px solid var(--color-border);
      border-radius:6px; padding:6px 8px;
      color:var(--color-text-primary); font-size:12px;
      outline:none; width:100%; box-sizing:border-box;
    }
    .field-input:focus { border-color:var(--color-accent); }

    /* ── Section divider / labels ────────────────────────────── */
    .section-divider {
      font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:0.08em; color:var(--color-text-secondary);
      margin:6px 0 0; padding-bottom:6px;
      border-bottom:1px solid var(--color-border);
    }
    .card-section-label {
      font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:0.06em; color:var(--color-text-secondary); margin:0 0 10px;
    }

    /* ── Advanced toggle ─────────────────────────────────────── */
    .advanced-toggle {
      font-size:11px; font-weight:500; color:var(--color-text-secondary);
      background:none; border:none; cursor:pointer; padding:4px 0; text-align:left;
    }
    .advanced-toggle:hover { color:var(--color-text-primary); }

    /* ── Strategy-style checkboxes ───────────────────────────── */
    .strategy-check {
      display:flex; align-items:center; gap:8px;
      padding:7px 10px; border-radius:6px;
      border:1px solid var(--color-border); background:var(--color-bg);
      cursor:pointer; font-size:12px; color:var(--color-text-secondary);
      transition:border-color 100ms, color 100ms;
    }
    .strategy-check.active {
      border-color:var(--color-accent); color:var(--color-text-primary);
      background:color-mix(in srgb, var(--color-accent) 8%, transparent);
    }
    .strategy-check input { display:none; }

    /* ── Setup layout ────────────────────────────────────────── */
    .setup-body {
      display:grid; grid-template-columns:320px 1fr;
      flex:1; overflow:hidden;
    }
    .config-panel {
      border-radius:0; border-top:none; border-bottom:none; border-left:none;
      padding:16px; overflow-y:auto;
      display:flex; flex-direction:column; gap:10px;
    }
    .quota-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
    .setup-right { padding:16px; overflow-y:auto; }

    /* Participants editor */
    .participants-list { display:flex; flex-direction:column; gap:6px; }
    .participants-list-header {
      display:grid; grid-template-columns:1fr 80px; gap:8px;
      font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:0.06em; color:var(--color-text-secondary); padding:0 2px;
    }
    .participant-edit-row { display:grid; grid-template-columns:1fr 80px; gap:8px; }
    .budget-input { text-align:right; }

    /* ── Live layout ─────────────────────────────────────────── */
    /* Price index strip */
    .price-strip {
      display:flex; gap:16px; padding:8px 24px;
      border-bottom:1px solid var(--color-border); flex-shrink:0;
      overflow-x:auto; background:var(--color-surface);
    }
    .price-role-group { display:flex; align-items:center; gap:6px; }
    .price-role-label { font-size:11px; font-weight:700; }
    .price-chip {
      font-size:11px; font-family:var(--font-mono); font-weight:500;
      padding:1px 6px; border-radius:4px; border:1px solid;
      background:color-mix(in srgb, currentColor 8%, transparent);
    }

    .auction-body {
      display:grid; grid-template-columns:260px 1fr;
      flex:1; overflow:hidden;
    }

    /* Participants panel */
    .participants-panel {
      border-right:1px solid var(--color-border);
      padding:12px; overflow-y:auto;
      display:flex; flex-direction:column; gap:6px;
    }
    .panel-heading {
      font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:0.08em; color:var(--color-text-secondary);
      margin:0 0 6px; padding-bottom:6px; border-bottom:1px solid var(--color-border);
    }

    .participant-card {
      padding:10px; border-radius:8px;
      border:1px solid var(--color-border); background:var(--color-surface);
      display:flex; flex-direction:column; gap:5px;
    }
    .participant-header {
      display:flex; justify-content:space-between; align-items:baseline;
    }
    .participant-name {
      font-size:12px; font-weight:600; color:var(--color-text-primary);
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:130px;
    }
    .participant-budget { font-size:12px; font-weight:700; font-variant-numeric:tabular-nums; }

    .budget-bar {
      height:3px; border-radius:9999px; background:var(--color-border); overflow:hidden;
    }
    .budget-bar-fill { height:100%; border-radius:9999px; transition:width 300ms ease; }

    .role-chips { display:flex; flex-wrap:wrap; gap:3px; }
    .role-chip {
      font-size:10px; font-weight:600; padding:1px 5px;
      border-radius:4px; border:1px solid; font-family:var(--font-mono);
      background:color-mix(in srgb, currentColor 8%, transparent);
    }
    .empty-squad { font-size:11px; color:var(--color-text-secondary); }

    /* Main area */
    .auction-main {
      display:flex; flex-direction:column; gap:0;
      overflow-y:auto; padding:16px; gap:16px;
    }
    .action-row { display:grid; grid-template-columns:1fr 1fr; gap:16px; }

    /* Lookup + pool autocomplete */
    .lookup-row { display:flex; gap:8px; align-items:center; margin-bottom:0; }

    .pool-autocomplete { position:relative; }
    .pool-dropdown {
      position:absolute; top:calc(100% + 4px); left:0; right:0; z-index:100;
      background:var(--color-surface-raised); border:1px solid var(--color-border);
      border-radius:8px; list-style:none; margin:0; padding:4px 0;
      max-height:220px; overflow-y:auto;
      box-shadow:0 8px 24px rgba(0,0,0,0.4);
    }
    .pool-option {
      display:flex; align-items:center; justify-content:space-between; gap:8px;
      padding:7px 12px; cursor:pointer; transition:background 80ms;
    }
    .pool-option:hover { background:var(--color-surface); }
    .pool-name { font-size:12px; font-weight:600; color:var(--color-text-primary); }
    .pool-meta {
      font-size:11px; color:var(--color-text-secondary);
      display:flex; align-items:center; gap:5px; flex-shrink:0;
    }

    .projection-row {
      display:flex; align-items:center; gap:8px; margin-top:10px;
      padding:8px 10px; border-radius:6px; background:var(--color-surface);
      border:1px solid var(--color-border);
    }
    .proj-label { font-size:11px; color:var(--color-text-secondary); flex:1; }
    .proj-price { font-size:18px; font-weight:700; color:var(--color-text-primary); font-variant-numeric:tabular-nums; }

    .alternatives-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:10px; }
    .alt-card {
      padding:8px; border-radius:6px; background:var(--color-surface);
      border:1px solid var(--color-border);
    }
    .alt-label { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--color-text-secondary); margin:0 0 3px; }
    .alt-name { font-size:12px; font-weight:600; color:var(--color-text-primary); margin:0 0 2px; }
    .alt-meta { font-size:11px; color:var(--color-text-secondary); margin:0; }
    .alt-none { font-size:11px; color:var(--color-text-secondary); margin:0; grid-column:span 2; }

    /* Record */
    .record-actions { display:flex; gap:8px; margin-top:4px; }
    .record-actions .run-btn { flex:1; }
    .record-actions .secondary-btn { flex-shrink:0; }

    .inline-error { font-size:11px; color:#EF4444; margin:6px 0 0; }
    .inline-rejection {
      padding:8px 10px; border-radius:6px; margin-top:4px;
      border:1px solid #EF4444;
      background:color-mix(in srgb,#EF4444 8%,transparent);
    }
    .rejection-code {
      font-family:var(--font-mono); font-size:11px; font-weight:700;
      color:#EF4444; display:block; margin:0 0 2px;
    }
    .rejection-msg { font-size:11px; color:var(--color-text-secondary); margin:0; }

    /* History */
    .history-card { padding:0; overflow:hidden; }
    .history-card .card-section-label { padding:12px 14px 0; }
    .history-table-wrap { max-height:280px; overflow-y:auto; }
    .empty-history {
      padding:24px; text-align:center;
      font-size:13px; color:var(--color-text-secondary);
    }

    /* Tier badge */
    .tier-badge {
      font-size:10px; font-weight:700; padding:1px 6px;
      border-radius:4px; border:1px solid;
      background:color-mix(in srgb, currentColor 8%, transparent);
      font-family:var(--font-mono); flex-shrink:0;
    }

    /* Squad table (reused from optimizer) */
    .squad-table { width:100%; border-collapse:collapse; font-size:12px; }
    .squad-table thead th {
      position:sticky; top:0; z-index:1;
      background:var(--color-surface); padding:8px 12px; text-align:left;
      font-size:10px; font-weight:600; text-transform:uppercase;
      letter-spacing:0.05em; color:var(--color-text-secondary);
      border-bottom:1px solid var(--color-border);
    }
    .squad-table thead th.num { text-align:right; }
    .squad-table tbody tr {
      border-bottom:1px solid var(--color-border); transition:background 100ms;
    }
    .squad-table tbody tr:hover { background:var(--color-surface-raised); }
    .squad-table tbody td { padding:7px 12px; color:var(--color-text-primary); vertical-align:middle; }
    .squad-table .num { text-align:right; font-variant-numeric:tabular-nums; }
    .seq { color:var(--color-text-secondary); font-variant-numeric:tabular-nums; width:32px; }
    .player-name { font-weight:500; margin:0; }
    .team-name { font-size:11px; color:var(--color-text-secondary); margin:0; }
    .role-badge {
      display:inline-flex; align-items:center; justify-content:center;
      width:28px; padding:1px 0; border-radius:4px;
      border:1px solid; font-size:10px; font-weight:700;
    }
    .faded { color:var(--color-text-secondary); }
    .accent { color:var(--color-accent); font-weight:600; }
  `],
})
export class AuctionComponent {
  private readonly auctionService = inject(AuctionService);
  private readonly quotationService = inject(QuotationService);

  readonly allRoles: readonly AuctionRole[] = AUCTION_ROLES;
  readonly allTiers: readonly AuctionTier[] = ['LOW', 'MID', 'TOP'];

  // ── Setup form state (plain properties — bound via (change) events) ──
  seasonStart = 2024;
  numParticipants = 8;
  defaultBudget = 500;
  showAdvanced = false;
  useInflationBaseline = true;
  referenceBudget = 300;
  budgetInitial = 300;
  roleQuotas: Partial<Record<AuctionRole, number>> = { P: 3, D: 8, C: 8, A: 6 };
  alpha = 0.3;
  spilloverAdj = 0.1;
  spilloverCross = 0.05;
  lowCostPercentile = 0.3;
  minIndex = 0.5;
  maxIndex = 2.0;
  tierLow = 0.3;
  tierTop = 0.7;

  private readonly destroyRef = inject(DestroyRef);

  // ── Async signals ────────────────────────────────────────────────────
  readonly seasons = signal<number[]>([]);
  readonly participants = signal<AuctionParticipantSetup[]>(makeParticipants(8, 500));
  readonly starting = signal(false);
  readonly initError = signal<string | null>(null);

  readonly sessionId = signal<string | null>(null);
  readonly summary = signal<AuctionSummary | null>(null);
  readonly summaryLoading = signal(false);

  readonly projection = signal<ProjectionResponse | null>(null);
  readonly altResult = signal<AlternativesResponse | null>(null);
  readonly lookupLoading = signal(false);
  readonly lookupError = signal<string | null>(null);

  readonly recordLoading = signal(false);
  readonly recordError = signal<string | null>(null);
  readonly recordRejectionCode = signal<string | null>(null);
  readonly undoLoading = signal(false);

  // ── Pool autocomplete ─────────────────────────────────────────────────
  readonly poolSuggestions = signal<AuctionPlayerSummary[]>([]);
  readonly poolOpen = signal(false);
  private readonly poolQuery$ = new Subject<string>();

  // ── Live form state (plain properties) ───────────────────────────────
  lookupQuery = '';    // display text in lookup input
  lookupId = '';       // resolved playerId
  recordPlayerId = '';
  recordPlayerName = ''; // display text in record input
  recordWinnerId = '';
  recordPrice = 1;

  // ── Initial budgets map (for budget-bar computation) ──────────────────
  private readonly initialBudgets = new Map<string, number>();

  readonly reversedAssignments = computed(() =>
    [...(this.summary()?.assignments ?? [])].reverse(),
  );

  constructor() {
    this.quotationService.getSeasons().subscribe({
      next: s => {
        const sorted = [...s].sort((a, b) => b - a);
        this.seasons.set(sorted);
        if (sorted.length) this.seasonStart = sorted[0];
      },
      error: () => this.seasons.set([2025, 2024, 2023]),
    });

    // Pool autocomplete: debounce query → call pool endpoint
    this.poolQuery$.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      switchMap(q => {
        const sid = this.sessionId();
        if (!sid) return [];
        return this.auctionService.pool(sid, q);
      }),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: items => { this.poolSuggestions.set(items); this.poolOpen.set(items.length > 0); },
      error: () => { this.poolSuggestions.set([]); this.poolOpen.set(false); },
    });
  }

  // ── Setup helpers ─────────────────────────────────────────────────────

  resizeParticipants(): void {
    this.participants.set(makeParticipants(this.numParticipants, this.defaultBudget, this.participants()));
  }

  updateName(i: number, name: string): void {
    this.participants.update(arr => {
      const next = [...arr];
      next[i] = { ...next[i], displayName: name };
      return next;
    });
  }

  updateBudget(i: number, budget: number): void {
    this.participants.update(arr => {
      const next = [...arr];
      next[i] = { ...next[i], budgetInitial: budget };
      return next;
    });
  }

  /** Propaga "Budget each" a tutti i participants. */
  applyDefaultBudget(): void {
    this.participants.update(arr =>
      arr.map(p => ({ ...p, budgetInitial: this.defaultBudget })),
    );
  }

  // ── Session init ──────────────────────────────────────────────────────

  startAuction(): void {
    this.starting.set(true);
    this.initError.set(null);
    this._cacheInitialBudgets(this.participants());

    this.auctionService.init({
      seasonStart: this.seasonStart,
      participants: this.participants(),
      config: {
        numParticipants: this.numParticipants,
        roleQuotas: this.roleQuotas,
        marketDriftConfig: {
          alpha: this.alpha,
          spilloverAdjacentTier: this.spilloverAdj,
          spilloverCrossRole: this.spilloverCross,
          minIndex: this.minIndex,
          maxIndex: this.maxIndex,
          tierThresholds: [this.tierLow, this.tierTop],
        },
        alternativesConfig: { lowCostPercentile: this.lowCostPercentile },
        useInflationBaseline: this.useInflationBaseline,
        referenceBudget: this.referenceBudget,
        budgetInitial: this.budgetInitial,
      },
    }).subscribe({
      next: res => {
        this.sessionId.set(res.sessionId);
        this.starting.set(false);
        this.refreshSummary();
      },
      error: err => {
        this.initError.set(err.error?.detail ?? 'Failed to start session');
        this.starting.set(false);
      },
    });
  }

  onResumeFile(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const payload = JSON.parse(e.target!.result as string);
        this.starting.set(true);
        this.initError.set(null);
        this.auctionService.deserialize({ payload }).subscribe({
          next: res => {
            this.sessionId.set(res.sessionId);
            this.starting.set(false);
            this.refreshSummary();
          },
          error: err => {
            this.initError.set(err.error?.detail ?? 'Failed to resume session');
            this.starting.set(false);
          },
        });
      } catch {
        this.initError.set('Invalid save file — must be JSON');
      }
    };
    reader.readAsText(file);
    // Reset input so same file can be re-selected
    (event.target as HTMLInputElement).value = '';
  }

  // ── Live actions ──────────────────────────────────────────────────────

  onPoolQueryChange(q: string): void {
    this.poolQuery$.next(q);
    if (!q.trim()) { this.poolSuggestions.set([]); this.poolOpen.set(false); }
  }

  selectPoolPlayer(p: AuctionPlayerSummary): void {
    this.lookupId = p.playerId;
    this.lookupQuery = p.name;
    // pre-fill record card too
    this.recordPlayerId = p.playerId;
    this.recordPlayerName = `${p.name} (${p.role} · ${p.realTeam})`;
    this.poolOpen.set(false);
    this.poolSuggestions.set([]);
    this.lookupPlayer(p.playerId);
  }

  lookupPlayer(playerId = this.lookupId): void {
    const sid = this.sessionId();
    if (!sid || !playerId) return;
    this.lookupLoading.set(true);
    this.lookupError.set(null);
    this.projection.set(null);
    this.altResult.set(null);

    forkJoin({
      proj: this.auctionService.projection(sid, playerId),
      alt: this.auctionService.alternatives(sid, playerId),
    }).subscribe({
      next: ({ proj, alt }) => {
        this.projection.set(proj);
        this.altResult.set(alt);
        this.lookupLoading.set(false);
      },
      error: err => {
        this.lookupError.set(err.error?.detail ?? 'Player not found');
        this.lookupLoading.set(false);
      },
    });
  }

  submitRecord(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.recordLoading.set(true);
    this.recordError.set(null);
    this.recordRejectionCode.set(null);

    this.auctionService.record(sid, {
      playerId: this.recordPlayerId,
      winnerParticipantId: this.recordWinnerId,
      finalPrice: this.recordPrice,
    }).subscribe({
      next: res => {
        if (!res.success) {
          this.recordError.set(res.rejectionReason ?? 'Assignment rejected');
          this.recordRejectionCode.set(res.rejectionCode ?? null);
        } else {
          this.recordPlayerId = '';
          this.recordPlayerName = '';
          this.recordWinnerId = '';
          this.recordPrice = 1;
          this.recordError.set(null);
          this.projection.set(null);
          this.altResult.set(null);
          this.lookupId = '';
          this.lookupQuery = '';
          this.refreshSummary();
        }
        this.recordLoading.set(false);
      },
      error: err => {
        this.recordError.set(err.error?.detail ?? 'Server error');
        this.recordLoading.set(false);
      },
    });
  }

  undoLast(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.undoLoading.set(true);
    this.auctionService.undo(sid).subscribe({
      next: () => { this.undoLoading.set(false); this.refreshSummary(); },
      error: () => this.undoLoading.set(false),
    });
  }

  saveToFile(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.auctionService.serialize(sid).subscribe({
      next: res => {
        const blob = new Blob([JSON.stringify(res.payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `auction_${sid.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      },
    });
  }

  endSession(): void {
    if (!confirm('End this auction session? The session will be deleted.')) return;
    const sid = this.sessionId();
    if (!sid) return;
    this.auctionService.discard(sid).subscribe({
      next: () => this._resetLiveState(),
      error: () => this._resetLiveState(),
    });
  }

  refreshSummary(): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.summaryLoading.set(true);
    this.auctionService.summary(sid).subscribe({
      next: s => {
        this.summary.set(s);
        this.summaryLoading.set(false);
        // Populate initialBudgets from setup participants if not already set
        if (this.initialBudgets.size === 0) {
          s.participants.forEach(p => {
            if (!this.initialBudgets.has(p.participantId)) {
              this.initialBudgets.set(p.participantId, p.budgetResidual);
            }
          });
        }
      },
      error: () => this.summaryLoading.set(false),
    });
  }

  // ── Template helpers ──────────────────────────────────────────────────

  roleColor(role: string): string { return ROLE_COLOR[role] ?? 'var(--color-text-secondary)'; }
  tierColor(tier: AuctionTier): string { return TIER_COLOR[tier]; }

  budgetPercent(p: AuctionParticipantState): number {
    const initial = this.initialBudgets.get(p.participantId) ?? p.budgetResidual;
    if (initial === 0) return 0;
    return Math.max(0, Math.min(100, (p.budgetResidual / initial) * 100));
  }

  budgetColor(p: AuctionParticipantState): string {
    const pct = this.budgetPercent(p) / 100;
    if (pct > 0.4) return 'var(--color-text-primary)';
    if (pct > 0.2) return '#F59E0B';
    return '#EF4444';
  }

  winnerName(participantId: string): string {
    return this.summary()?.participants.find(p => p.participantId === participantId)?.displayName
      ?? participantId;
  }

  // ── Private ───────────────────────────────────────────────────────────

  private _cacheInitialBudgets(participants: AuctionParticipantSetup[]): void {
    this.initialBudgets.clear();
    participants.forEach(p => this.initialBudgets.set(p.participantId, p.budgetInitial));
  }

  private _resetLiveState(): void {
    this.sessionId.set(null);
    this.summary.set(null);
    this.projection.set(null);
    this.altResult.set(null);
    this.poolSuggestions.set([]);
    this.poolOpen.set(false);
    this.lookupId = '';
    this.lookupQuery = '';
    this.recordPlayerId = '';
    this.recordPlayerName = '';
    this.recordWinnerId = '';
    this.recordPrice = 1;
    this.initialBudgets.clear();
  }
}
