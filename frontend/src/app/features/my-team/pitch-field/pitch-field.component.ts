import { Component, computed, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { pitchSlotsForFormation } from '../../../core/constants/pitch-coordinates';
import { SlotAssignment } from '../../../core/models/my-team.models';

export interface PitchPlayerView {
  slotLabel: string;
  playerName: string;
  expectedScore: number;
  starterProbability?: number;
}

@Component({
  selector: 'app-pitch-field',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="w-full max-w-md mx-auto">
      <svg
        viewBox="0 0 100 110"
        class="w-full h-auto rounded-xl"
        role="img"
        [attr.aria-label]="'Campo ' + (formation() || '')"
      >
        <!-- pitch background -->
        <rect x="0" y="0" width="100" height="100" rx="2" fill="#1a5c38" />
        <rect x="1.5" y="1.5" width="97" height="97" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="0.4" />
        <!-- halfway -->
        <line x1="1.5" y1="50" x2="98.5" y2="50" stroke="rgba(255,255,255,0.3)" stroke-width="0.35" />
        <circle cx="50" cy="50" r="9" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="0.35" />
        <circle cx="50" cy="50" r="0.8" fill="rgba(255,255,255,0.4)" />
        <!-- boxes -->
        <rect x="30" y="1.5" width="40" height="14" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="0.35" />
        <rect x="38" y="1.5" width="24" height="6" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="0.35" />
        <rect x="30" y="84.5" width="40" height="14" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="0.35" />
        <rect x="38" y="92.5" width="24" height="6" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="0.35" />

        @for (p of placed(); track p.slotLabel + p.playerName) {
          <g [attr.transform]="'translate(' + p.x + ' ' + p.y + ')'">
            <circle r="5.2" fill="rgba(15,23,42,0.85)" stroke="#34d399" stroke-width="0.45" />
            <text
              text-anchor="middle"
              dy="0.35"
              font-size="2.4"
              fill="#ecfdf5"
              font-family="system-ui, sans-serif"
              font-weight="600"
            >
              {{ shortName(p.playerName) }}
            </text>
            <text
              text-anchor="middle"
              y="8.2"
              font-size="2"
              fill="rgba(236,253,245,0.85)"
              font-family="system-ui, sans-serif"
            >
              {{ p.expectedScore | number: '1.1-1' }}
            </text>
          </g>
        }

        <text
          x="50"
          y="106"
          text-anchor="middle"
          font-size="3.2"
          fill="currentColor"
          opacity="0.7"
          font-family="system-ui, sans-serif"
        >
          {{ formation() || '—' }}
          @if (score() != null) {
            <ng-container> · {{ score() | number: '1.1-1' }}</ng-container>
          }
        </text>
      </svg>
    </div>
  `,
})
export class PitchFieldComponent {
  readonly formation = input<string | null>(null);
  readonly players = input<PitchPlayerView[]>([]);
  readonly score = input<number | null>(null);

  readonly placed = computed(() => {
    const slots = pitchSlotsForFormation(this.formation());
    const players = this.players();
    if (!players.length) {
      return [] as Array<PitchPlayerView & { x: number; y: number }>;
    }

    const used = new Set<number>();
    const result: Array<PitchPlayerView & { x: number; y: number }> = [];

    for (const p of players) {
      let idx = slots.findIndex(
        (s, i) => !used.has(i) && this.slotMatch(s.key, p.slotLabel),
      );
      if (idx < 0) {
        idx = slots.findIndex((_, i) => !used.has(i));
      }
      if (idx < 0) continue;
      used.add(idx);
      result.push({ ...p, x: slots[idx].x, y: slots[idx].y });
    }
    return result;
  });

  shortName(name: string): string {
    const parts = name.trim().split(/\s+/);
    if (parts.length === 1) return parts[0].slice(0, 8);
    return parts[parts.length - 1].slice(0, 8);
  }

  private slotMatch(coordKey: string, slotLabel: string): boolean {
    if (coordKey === slotLabel) return true;
    // Por
    if (coordKey === 'Por' && (slotLabel === 'Por' || slotLabel.toLowerCase().includes('por'))) {
      return true;
    }
    // strip instance suffix #1
    const base = (s: string) => s.replace(/#\d+$/, '');
    if (base(coordKey) === base(slotLabel)) return true;
    // partial role overlap e.g. W/A vs W
    const roles = (s: string) =>
      base(s)
        .split('/')
        .map((x) => x.trim().toLowerCase());
    const a = new Set(roles(coordKey));
    const b = roles(slotLabel);
    return b.some((r) => a.has(r));
  }
}

/** Map API starting XI to pitch views. */
export function toPitchPlayers(
  xi: SlotAssignment[] | undefined | null,
): PitchPlayerView[] {
  if (!xi?.length) return [];
  return xi.map((s) => ({
    slotLabel: s.slotLabel,
    playerName: s.playerName,
    expectedScore: s.expectedScore,
    starterProbability: s.starterProbability,
  }));
}
