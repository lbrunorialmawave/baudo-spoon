import {
  Component, input, inject, PLATFORM_ID, DestroyRef, afterRenderEffect,
  ElementRef, viewChild,
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import * as d3 from 'd3';
import { QuotationRoleAggregate } from '../../../../core/models/quotations.models';

const ROLE_COLORS: Record<string, string> = {
  GK:  '#F59E0B',
  DEF: '#22C55E',
  MID: '#3B82F6',
  FWD: '#EF4444',
};
const ROLES = ['GK', 'DEF', 'MID', 'FWD'] as const;

@Component({
  selector: 'app-quotation-chart',
  standalone: true,
  template: `
    <div class="relative w-full" #container>
      <svg #svgEl class="block w-full" aria-label="Average auction price by role per season" role="img"></svg>
      <!-- Legend -->
      <div class="mt-2 flex items-center gap-4 justify-center">
        @for (role of roles; track role) {
          <div class="flex items-center gap-1.5 text-xs" style="color:var(--color-text-secondary)">
            <span class="inline-block w-3 h-3 rounded-sm" [style.background]="roleColors[role]"></span>
            {{ role }}
          </div>
        }
      </div>
    </div>
  `,
  styles: [':host { display: block; }'],
})
export class QuotationChartComponent {
  readonly data = input.required<QuotationRoleAggregate[]>();

  private readonly containerRef = viewChild.required<ElementRef<HTMLDivElement>>('container');
  private readonly svgRef = viewChild.required<ElementRef<SVGSVGElement>>('svgEl');
  private readonly platformId = inject(PLATFORM_ID);
  private readonly destroyRef = inject(DestroyRef);
  private resizeObserver: ResizeObserver | null = null;

  readonly roles = ROLES;
  readonly roleColors = ROLE_COLORS;

  constructor() {
    afterRenderEffect(() => {
      if (!isPlatformBrowser(this.platformId)) return;
      const data = this.data();
      if (!data.length) return;
      this.draw(data);
    });
    this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
  }

  private draw(data: QuotationRoleAggregate[]): void {
    this.resizeObserver?.disconnect();

    const container = this.containerRef().nativeElement;
    const svgEl = this.svgRef().nativeElement;

    const M = { top: 16, right: 16, bottom: 36, left: 44 };
    const W = Math.max(container.clientWidth || 480, 320);
    const H = 240;
    const iW = W - M.left - M.right;
    const iH = H - M.top - M.bottom;

    const seasons = [...new Set(data.map(d => d.seasonStart))].sort();
    const x0 = d3.scaleBand().domain(seasons.map(String)).range([0, iW]).padding(0.25);
    const x1 = d3.scaleBand().domain([...ROLES]).range([0, x0.bandwidth()]).padding(0.08);
    const maxVal = d3.max(data, d => d.avgQtA) ?? 100;
    const y = d3.scaleLinear().domain([0, maxVal * 1.1]).range([iH, 0]);

    const sel = d3.select(svgEl);
    sel.selectAll('*').remove();
    sel.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);

    const root = sel.append('g').attr('transform', `translate(${M.left},${M.top})`);
    const tc = 'hsl(220,10%,55%)';
    const gc = 'hsl(220,15%,18%)';

    // Gridlines
    root.append('g')
      .call((d3.axisLeft(y).tickSize(-iW).ticks(4) as d3.Axis<d3.NumberValue>).tickFormat(() => ''))
      .call(g => {
        g.select('.domain').remove();
        g.selectAll('.tick line').attr('stroke', gc).attr('stroke-dasharray', '3 5');
      });

    // Axes
    root.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(x0).tickSize(0))
      .call(g => {
        g.select('.domain').attr('stroke', gc);
        g.selectAll('text').attr('fill', tc).attr('font-size', '11');
      });

    root.append('g')
      .call(d3.axisLeft(y).ticks(4).tickFormat(d => String(d)))
      .call(g => {
        g.select('.domain').attr('stroke', gc);
        g.selectAll('.tick line').remove();
        g.selectAll('text').attr('fill', tc).attr('font-size', '11');
      });

    // Bars grouped by season
    const seasonGroups = root.selectAll('.season-group')
      .data(seasons)
      .join('g')
      .attr('class', 'season-group')
      .attr('transform', s => `translate(${x0(String(s)) ?? 0},0)`);

    seasonGroups.each(function(season) {
      const g = d3.select(this);
      ROLES.forEach(role => {
        const row = data.find(d => d.seasonStart === season && d.role === role);
        if (!row) return;
        g.append('rect')
          .attr('x', x1(role) ?? 0)
          .attr('y', y(row.avgQtA))
          .attr('width', x1.bandwidth())
          .attr('height', iH - y(row.avgQtA))
          .attr('fill', ROLE_COLORS[role])
          .attr('fill-opacity', 0.85)
          .attr('rx', 2);
      });
    });

    // Resize
    let rafId = 0;
    this.resizeObserver = new ResizeObserver(() => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => this.draw(data));
    });
    this.resizeObserver.observe(container);
  }
}
