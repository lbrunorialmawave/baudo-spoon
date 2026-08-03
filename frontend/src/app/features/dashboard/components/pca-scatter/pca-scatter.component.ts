import {
  Component, input, output, ElementRef, viewChild,
  inject, signal, computed, DestroyRef, PLATFORM_ID, afterRenderEffect,
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import * as d3 from 'd3';
import { PlayerCluster } from '../../../../core/models/api.models';
import { CLUSTER_COLORS } from '../../../../core/constants/cluster-colors';

interface TooltipState {
  x: number;
  y: number;
  player: PlayerCluster;
}

@Component({
  selector: 'app-pca-scatter',
  standalone: true,
  template: `
    <div class="relative w-full" #container>
      @if (tooltip(); as tt) {
        <div
          class="pointer-events-none absolute z-10 rounded-lg border px-3 py-2.5 text-xs shadow-2xl"
          [style]="tooltipStyle(tt)"
          role="tooltip"
        >
          <p class="font-semibold">{{ tt.player.playerName }}</p>
          <p class="mt-0.5" style="color:var(--color-text-secondary)">
            {{ tt.player.teamName ?? '—' }} · {{ tt.player.canonicalRole ?? '—' }}
          </p>
          <div class="mt-1.5 flex items-center gap-1.5">
            <span
              class="h-2 w-2 shrink-0 rounded-full"
              [style.background]="clusterColor(tt.player.clusterId)"
            ></span>
            <span style="color:var(--color-text-secondary)">
              Cluster {{ tt.player.clusterId }}
            </span>
            @if (tt.player.predictedFantavoto !== null) {
              <span class="ml-1 font-semibold text-brand-400">
                ★ {{ tt.player.predictedFantavoto!.toFixed(2) }}
              </span>
            }
          </div>
        </div>
      }

      <svg
        #svgEl
        class="scatter-svg block w-full"
        [attr.aria-label]="ariaLabel()"
        role="img"
      ></svg>
    </div>
  `,
  styles: [`
    :host { display: block; }
    /* d3-zoom owns pan/pinch on this element; without this the browser's
       own scroll gesture fights the zoom behavior on touch devices. */
    .scatter-svg { touch-action: none; }
  `],
})
export class PcaScatterComponent {
  readonly players     = input.required<PlayerCluster[]>();
  readonly playerSelected = output<PlayerCluster>();

  private readonly containerRef = viewChild.required<ElementRef<HTMLDivElement>>('container');
  private readonly svgRef       = viewChild.required<ElementRef<SVGSVGElement>>('svgEl');

  private readonly platformId = inject(PLATFORM_ID);
  private readonly destroyRef = inject(DestroyRef);

  readonly tooltip = signal<TooltipState | null>(null);

  readonly ariaLabel = computed(() => {
    const players = this.players();
    if (!players?.length) return 'Player cluster scatter plot';
    const clusterCount = new Set(players.map(p => p.clusterId)).size;
    return `Scatter plot: ${players.length} giocatori in ${clusterCount} cluster`;
  });

  private resizeObserver: ResizeObserver | null = null;

  constructor() {
    afterRenderEffect(() => {
      if (!isPlatformBrowser(this.platformId)) return;
      const players = this.players();
      if (!players?.length) return;
      // Filter out players without PCA coords
      const valid = players.filter(p => p.pca0 !== null && p.pca1 !== null);
      if (!valid.length) return;
      this.draw(valid);
    });

    this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
  }

  clusterColor(id: number): string {
    return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
  }

  tooltipStyle(tt: TooltipState): string {
    return [
      `left:${tt.x}px`,
      `top:${tt.y}px`,
      `transform:translate(-50%,calc(-100% - 10px))`,
      `background:var(--color-surface-raised)`,
      `border-color:var(--color-border)`,
      `color:var(--color-text-primary)`,
    ].join(';');
  }

  private draw(players: PlayerCluster[]): void {
    this.resizeObserver?.disconnect();

    const container = this.containerRef().nativeElement;
    const svgEl     = this.svgRef().nativeElement;

    const M  = { top: 24, right: 152, bottom: 48, left: 52 };
    const W  = Math.max(container.clientWidth || 640, 320);
    const H  = 400;
    const iW = W - M.left - M.right;
    const iH = H - M.top  - M.bottom;

    // Safe numeric accessors (pca0/pca1 are non-null after the filter above)
    const px = (d: PlayerCluster) => d.pca0!;
    const py = (d: PlayerCluster) => d.pca1!;

    // ── Scales ─────────────────────────────────────────────
    const xExt = d3.extent(players, px) as [number, number];
    const yExt = d3.extent(players, py) as [number, number];
    const xPad = (xExt[1] - xExt[0]) * 0.1 || 0.5;
    const yPad = (yExt[1] - yExt[0]) * 0.1 || 0.5;

    const xScale = d3.scaleLinear()
      .domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, iW]);
    const yScale = d3.scaleLinear()
      .domain([yExt[0] - yPad, yExt[1] + yPad]).range([iH, 0]);

    // ── SVG root ──────────────────────────────────────────
    const sel = d3.select(svgEl);
    sel.selectAll('*').remove();
    sel.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);

    sel.append('defs').append('clipPath').attr('id', 'scatter-clip')
      .append('rect').attr('width', iW).attr('height', iH);

    const root = sel.append('g').attr('transform', `translate(${M.left},${M.top})`);

    // ── Grid ──────────────────────────────────────────────
    const gc = 'hsl(220,15%,18%)';

    root.append('g')
      .call((d3.axisLeft(yScale).tickSize(-iW).ticks(5) as d3.Axis<d3.NumberValue>)
        .tickFormat(() => ''))
      .call(g => {
        g.select('.domain').remove();
        g.selectAll('.tick line').attr('stroke', gc).attr('stroke-dasharray', '3 5');
      });

    root.append('g').attr('transform', `translate(0,${iH})`)
      .call((d3.axisBottom(xScale).tickSize(-iH).ticks(5) as d3.Axis<d3.NumberValue>)
        .tickFormat(() => ''))
      .call(g => {
        g.select('.domain').remove();
        g.selectAll('.tick line').attr('stroke', gc).attr('stroke-dasharray', '3 5');
      });

    // ── Axes ──────────────────────────────────────────────
    const ac  = 'hsl(220,10%,40%)';
    const tc  = 'hsl(220,10%,58%)';
    const fmt = (d: d3.NumberValue) => Number(d).toFixed(1);

    root.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(fmt))
      .call(g => {
        g.select('.domain').attr('stroke', ac);
        g.selectAll('.tick line').attr('stroke', ac);
        g.selectAll('text').attr('fill', tc).attr('font-size', '11');
      });

    root.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(fmt))
      .call(g => {
        g.select('.domain').attr('stroke', ac);
        g.selectAll('.tick line').attr('stroke', ac);
        g.selectAll('text').attr('fill', tc).attr('font-size', '11');
      });

    root.append('text')
      .attr('x', iW / 2).attr('y', iH + 40)
      .attr('text-anchor', 'middle').attr('fill', tc).attr('font-size', '12')
      .text('PC 1');

    root.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(iH / 2)).attr('y', -40)
      .attr('text-anchor', 'middle').attr('fill', tc).attr('font-size', '12')
      .text('PC 2');

    // ── Dots ──────────────────────────────────────────────
    const dotsGroup = root.append('g').attr('clip-path', 'url(#scatter-clip)');

    dotsGroup.selectAll<SVGCircleElement, PlayerCluster>('circle')
      .data(players, d => d.playerFotmobId ?? d.playerName)
      .join('circle')
      .attr('cx', d => xScale(px(d)))
      .attr('cy', d => yScale(py(d)))
      .attr('r', 5.5)
      .attr('fill', d => CLUSTER_COLORS[d.clusterId % CLUSTER_COLORS.length])
      .attr('fill-opacity', 0.82)
      .attr('stroke', 'hsl(220,15%,10%)')
      .attr('stroke-width', 1.2)
      .style('cursor', 'pointer')
      .on('mouseenter', (event: MouseEvent, d: PlayerCluster) => {
        d3.select(event.currentTarget as SVGCircleElement)
          .raise().transition().duration(80)
          .attr('r', 9).attr('fill-opacity', 1);

        const r    = (event.currentTarget as Element).getBoundingClientRect();
        const cBox = container.getBoundingClientRect();
        this.tooltip.set({ x: r.left - cBox.left + r.width / 2, y: r.top - cBox.top, player: d });
      })
      .on('mouseleave', (event: MouseEvent) => {
        d3.select(event.currentTarget as SVGCircleElement)
          .transition().duration(80)
          .attr('r', 5.5).attr('fill-opacity', 0.82);
        this.tooltip.set(null);
      })
      .on('touchstart', (event: TouchEvent, d: PlayerCluster) => {
        // Touch has no hover: surface the same preview mouseenter gives,
        // the tap that follows still opens the full player card via 'click'.
        const r    = (event.currentTarget as Element).getBoundingClientRect();
        const cBox = container.getBoundingClientRect();
        this.tooltip.set({ x: r.left - cBox.left + r.width / 2, y: r.top - cBox.top, player: d });
      })
      .on('click', (_: MouseEvent, d: PlayerCluster) => this.playerSelected.emit(d));

    // ── Zoom ──────────────────────────────────────────────
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.4, 12])
      .filter(event => !(event as MouseEvent).button)
      .on('zoom', ev => dotsGroup.attr('transform', ev.transform));

    sel.call(zoom).on('dblclick.zoom', null);

    // ── Legend ────────────────────────────────────────────
    const clusters = [...new Map(players.map(p => [p.clusterId, p])).values()]
      .sort((a, b) => a.clusterId - b.clusterId);

    const legend = root.append('g').attr('transform', `translate(${iW + 14}, 0)`);
    legend.append('text').attr('y', -4)
      .attr('font-size', '11').attr('font-weight', '600').attr('fill', tc)
      .text('Cluster');

    clusters.forEach((p, i) => {
      const row = legend.append('g').attr('transform', `translate(0,${i * 22 + 14})`);
      row.append('circle').attr('cx', 6).attr('cy', 0).attr('r', 6)
        .attr('fill', CLUSTER_COLORS[p.clusterId % CLUSTER_COLORS.length]);
      row.append('text').attr('x', 16).attr('y', 4)
        .attr('font-size', '11').attr('fill', tc)
        .text(`C${p.clusterId}`);
    });

    // ── Responsive resize ─────────────────────────────────
    let rafId = 0;
    this.resizeObserver = new ResizeObserver(() => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => this.draw(players));
    });
    this.resizeObserver.observe(container);
  }
}
