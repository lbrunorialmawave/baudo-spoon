/**
 * Relative pitch coordinates (x, y) in [0, 100] for Mantra modules.
 * Origin top-left; GK near bottom (y≈92); attack near top (y≈12).
 * Keys must match backend MANTRA formation labels.
 */

export interface PitchSlotCoord {
  /** Matches optimizer slot labels where possible (Por, Dc, W/A, …). */
  key: string;
  x: number;
  y: number;
}

/** Generic fallback layout by line counts derived from formation string. */
function layoutFromLabel(label: string): PitchSlotCoord[] {
  const parts = label.split('-').map((n) => parseInt(n, 10)).filter((n) => !Number.isNaN(n));
  const slots: PitchSlotCoord[] = [{ key: 'Por', x: 50, y: 92 }];
  if (parts.length === 0) return slots;

  const lineYs = [72, 52, 32, 14];
  // If 4 lines (e.g. 4-2-3-1), use four bands; if 3 lines, three bands.
  const ys =
    parts.length >= 4
      ? [70, 50, 30, 12]
      : parts.length === 3
        ? [68, 42, 16]
        : lineYs.slice(0, parts.length);

  parts.forEach((count, lineIdx) => {
    const y = ys[lineIdx] ?? 40;
    for (let i = 0; i < count; i++) {
      const x = count === 1 ? 50 : 12 + (i * (76 / Math.max(count - 1, 1)));
      slots.push({ key: `L${lineIdx}-${i}`, x, y });
    }
  });
  return slots;
}

/** Hand-tuned layouts for the most common modules (better visual balance). */
const TUNED: Record<string, PitchSlotCoord[]> = {
  '4-3-3': [
    { key: 'Por', x: 50, y: 92 },
    { key: 'Dd', x: 18, y: 72 },
    { key: 'Dc#1', x: 38, y: 74 },
    { key: 'Dc#2', x: 62, y: 74 },
    { key: 'Ds', x: 82, y: 72 },
    { key: 'M/C', x: 50, y: 52 },
    { key: 'C#1', x: 32, y: 48 },
    { key: 'C#2', x: 68, y: 48 },
    { key: 'W/A#1', x: 18, y: 22 },
    { key: 'A/Pc', x: 50, y: 14 },
    { key: 'W/A#2', x: 82, y: 22 },
  ],
  '3-5-2': [
    { key: 'Por', x: 50, y: 92 },
    { key: 'Dc#1', x: 30, y: 74 },
    { key: 'DC/B', x: 50, y: 76 },
    { key: 'Dc#2', x: 70, y: 74 },
    { key: 'E#1', x: 12, y: 52 },
    { key: 'M/C', x: 35, y: 48 },
    { key: 'C', x: 50, y: 50 },
    { key: 'C/T', x: 65, y: 48 },
    { key: 'E#2', x: 88, y: 52 },
    { key: 'A/Pc#1', x: 38, y: 16 },
    { key: 'A/Pc#2', x: 62, y: 16 },
  ],
  '3-4-3': [
    { key: 'Por', x: 50, y: 92 },
    { key: 'Dc#1', x: 30, y: 74 },
    { key: 'DC/B', x: 50, y: 76 },
    { key: 'Dc#2', x: 70, y: 74 },
    { key: 'E#1', x: 15, y: 50 },
    { key: 'M/C', x: 40, y: 48 },
    { key: 'C', x: 60, y: 48 },
    { key: 'E#2', x: 85, y: 50 },
    { key: 'W/A#1', x: 22, y: 20 },
    { key: 'A/Pc', x: 50, y: 12 },
    { key: 'W/A#2', x: 78, y: 20 },
  ],
  '4-4-2': [
    { key: 'Por', x: 50, y: 92 },
    { key: 'Dd', x: 18, y: 72 },
    { key: 'Dc#1', x: 38, y: 74 },
    { key: 'Dc#2', x: 62, y: 74 },
    { key: 'Ds', x: 82, y: 72 },
    { key: 'E/W', x: 15, y: 48 },
    { key: 'M/C', x: 38, y: 50 },
    { key: 'C', x: 62, y: 50 },
    { key: 'E', x: 85, y: 48 },
    { key: 'A/Pc#1', x: 38, y: 16 },
    { key: 'A/Pc#2', x: 62, y: 16 },
  ],
  '4-2-3-1': [
    { key: 'Por', x: 50, y: 92 },
    { key: 'Dd', x: 18, y: 74 },
    { key: 'Dc#1', x: 38, y: 76 },
    { key: 'Dc#2', x: 62, y: 76 },
    { key: 'Ds', x: 82, y: 74 },
    { key: 'M#1', x: 38, y: 56 },
    { key: 'M#2', x: 62, y: 56 },
    { key: 'W#1', x: 18, y: 34 },
    { key: 'T', x: 50, y: 32 },
    { key: 'W#2', x: 82, y: 34 },
    { key: 'A/Pc', x: 50, y: 12 },
  ],
};

export function pitchSlotsForFormation(formation: string | null | undefined): PitchSlotCoord[] {
  if (!formation) {
    return layoutFromLabel('4-3-3');
  }
  if (TUNED[formation]) {
    return TUNED[formation];
  }
  return layoutFromLabel(formation);
}

export const MANTRA_FORMATION_LABELS = [
  '3-4-3',
  '3-4-1-2',
  '3-4-2-1',
  '3-5-2',
  '3-5-1-1',
  '4-3-3',
  '4-3-1-2',
  '4-4-2',
  '4-1-4-1',
  '4-4-1-1',
  '4-2-3-1',
] as const;
