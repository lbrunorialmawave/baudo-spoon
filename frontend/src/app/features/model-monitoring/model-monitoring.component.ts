import {
  Component, DestroyRef, ElementRef, PLATFORM_ID,
  afterRenderEffect, computed, inject, signal, viewChild,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { isPlatformBrowser, DatePipe } from '@angular/common';
import * as d3 from 'd3';
import { ModelMetricsService } from '../../core/services/model-metrics.service';
import { MetricPoint, ModelRun } from '../../core/models/model-metrics.models';
import { SkeletonComponent } from '../../shared/components/skeleton/skeleton.component';
import { ErrorBoundaryComponent } from '../../shared/components/error-boundary/error-boundary.component';

@Component({
  selector: 'app-model-monitoring',
  standalone: true,
  imports: [SkeletonComponent, ErrorBoundaryComponent, DatePipe],
  template: `
    <div class="page-container">
      <header class="page-header">
        <div>
          <h1 class="page-title">Model Monitoring</h1>
          <p class="page-subtitle">ML pipeline run history and performance tracking</p>
        </div>
        @if (isDegraded()) {
          <span class="degraded-badge" role="alert" aria-live="polite">
            ⚠ Latest run degraded
          </span>
        }
      </header>

      @if (error()) {
        <app-error-boundary [message]="error()!" />
      } @else if (loading()) {
        <app-skeleton [height]="280" />
        <app-skeleton [height]="200" style="margin-top:1rem" />
      } @else {
        <!-- RMSE chart -->
        <section class="card chart-card">
          <h2 class="section-title">RMSE over time (test split)</h2>
          <div class="chart-wrap" #chartContainer>
            <svg #chartSvg class="chart-svg" role="img" aria-label="RMSE over time"></svg>
          </div>
        </section>

        <!-- Runs table -->
        <section class="card" style="margin-top:1.5rem">
          <h2 class="section-title">Pipeline runs</h2>
          <div class="table-scroll">
            <table class="runs-table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Model</th>
                  <th>Trained at</th>
                  <th>Season</th>
                  <th>Git</th>
                  <th>RMSE (test)</th>
                  <th>MAE (test)</th>
                  <th>R² (test)</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                @for (run of runs(); track run.run_id) {
                  <tr [class.degraded-row]="run.status === 'degraded'">
                    <td class="run-id-cell">{{ run.run_id }}</td>
                    <td>{{ run.model_name }}</td>
                    <td>{{ run.trained_at | date:'dd MMM yy HH:mm' }}</td>
                    <td>{{ run.season_start ?? '—' }}</td>
                    <td class="mono">{{ run.git_commit ?? '—' }}</td>
                    <td class="mono">{{ metricValue(run, 'rmse', 'test') }}</td>
                    <td class="mono">{{ metricValue(run, 'mae', 'test') }}</td>
                    <td class="mono">{{ metricValue(run, 'r2', 'test') }}</td>
                    <td>
                      <span class="status-badge" [attr.data-status]="run.status">
                        {{ run.status }}
                      </span>
                    </td>
                  </tr>
                } @empty {
                  <tr><td colspan="9" class="empty-row">No runs recorded yet.</td></tr>
                }
              </tbody>
            </table>
          </div>
        </section>
      }
    </div>
  `,
  styles: [`
    .page-container { padding: 1.5rem; max-width: 1200px; margin: 0 auto; }
    .page-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; }
    .page-title { font-size: 1.5rem; font-weight: 700; }
    .page-subtitle { color: var(--color-text-secondary); font-size: 0.875rem; }
    .card { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 0.75rem; padding: 1.25rem; }
    .section-title { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; }
    .chart-wrap { width: 100%; }
    .chart-svg { display: block; width: 100%; }
    .degraded-badge {
      background: #fef2f2; color: #dc2626; border: 1px solid #fecaca;
      padding: 0.375rem 0.75rem; border-radius: 0.5rem; font-size: 0.875rem; font-weight: 600;
    }
    .table-scroll { overflow-x: auto; }
    .runs-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    .runs-table th { text-align: left; padding: 0.5rem 0.75rem; color: var(--color-text-secondary); font-weight: 500; border-bottom: 1px solid var(--color-border); white-space: nowrap; }
    .runs-table td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--color-border); }
    .run-id-cell { font-family: monospace; font-size: 0.8rem; }
    .mono { font-family: monospace; }
    .degraded-row td { background: #fff7ed; }
    .empty-row { text-align: center; color: var(--color-text-secondary); padding: 2rem; }
    .status-badge { padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
    .status-badge[data-status="ok"] { background: #dcfce7; color: #16a34a; }
    .status-badge[data-status="degraded"] { background: #fef9c3; color: #a16207; }
    .status-badge[data-status="error"] { background: #fef2f2; color: #dc2626; }
  `],
})
export class ModelMonitoringComponent {
  private readonly svc = inject(ModelMetricsService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly platformId = inject(PLATFORM_ID);

  readonly chartContainer = viewChild<ElementRef<HTMLDivElement>>('chartContainer');
  readonly chartSvg = viewChild<ElementRef<SVGSVGElement>>('chartSvg');

  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly history = signal<MetricPoint[]>([]);
  readonly runs = signal<ModelRun[]>([]);

  readonly isDegraded = computed(() => this.runs()[0]?.status === 'degraded');

  constructor() {
    this.svc.getRuns(undefined, 50).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (res) => this.runs.set(res.items),
      error: (e) => this.error.set(e?.message ?? 'Failed to load runs'),
    });

    this.svc.getHistory('rmse', 'test').pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (pts) => {
        this.history.set(pts);
        this.loading.set(false);
      },
      error: (e) => {
        this.error.set(e?.message ?? 'Failed to load history');
        this.loading.set(false);
      },
    });

    // Re-draw chart whenever history signal changes (zoneless-safe).
    afterRenderEffect(() => {
      const pts = this.history();
      const svg = this.chartSvg()?.nativeElement;
      const wrap = this.chartContainer()?.nativeElement;
      if (!isPlatformBrowser(this.platformId) || !svg || !wrap || pts.length === 0) return;
      this._drawChart(svg, wrap, pts);
    });
  }

  metricValue(run: ModelRun, metric: string, split: string): string {
    const m = run.metrics?.find(x => x.metric === metric && x.split === split);
    return m ? m.value.toFixed(4) : '—';
  }

  private _drawChart(svg: SVGSVGElement, wrap: HTMLDivElement, pts: MetricPoint[]): void {
    const W = wrap.clientWidth || 800;
    const H = 220;
    const margin = { top: 20, right: 20, bottom: 40, left: 56 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    d3.select(svg).selectAll('*').remove();
    d3.select(svg).attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);

    const g = d3.select(svg).append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleTime()
      .domain(d3.extent(pts, d => new Date(d.trained_at)) as [Date, Date])
      .range([0, innerW]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(pts, d => d.metric_value)! * 1.1])
      .range([innerH, 0])
      .nice();

    g.append('g').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%d %b') as any))
      .selectAll('text').style('font-size', '11px');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d => d3.format('.3f')(d as number)))
      .selectAll('text').style('font-size', '11px');

    // Grid lines
    g.append('g').attr('class', 'grid')
      .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat('' as any))
      .selectAll('line').style('stroke', '#e5e7eb').style('stroke-dasharray', '3,3');
    g.select('.grid .domain').remove();

    // Line
    const line = d3.line<MetricPoint>()
      .x(d => x(new Date(d.trained_at)))
      .y(d => y(d.metric_value))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(pts)
      .attr('fill', 'none')
      .attr('stroke', 'var(--color-accent, #6366f1)')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Dots — degraded runs in red
    g.selectAll('circle')
      .data(pts)
      .join('circle')
      .attr('cx', d => x(new Date(d.trained_at)))
      .attr('cy', d => y(d.metric_value))
      .attr('r', 4)
      .attr('fill', d => d.status === 'degraded' ? '#ef4444' : 'var(--color-accent, #6366f1)')
      .attr('stroke', 'white')
      .attr('stroke-width', 1.5);

    // Y-axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2).attr('y', -44)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'var(--color-text-secondary)')
      .text('RMSE');
  }
}
