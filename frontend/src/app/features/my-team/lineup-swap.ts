import { LineupOptimizeResponse, SlotAssignment } from '../../core/models/my-team.models';

/** Roles on a slot assignment (Mantra OR-group). */
export function rolesOf(slot: SlotAssignment): Set<string> {
  const fromApi = slot.slotRoles?.length
    ? slot.slotRoles
    : slot.slotLabel.replace(/#\d+$/, '').split('/');
  return new Set(fromApi.map((r) => r.trim()).filter(Boolean));
}

/** True if bench player can fill the starter slot (role intersection). */
export function canSwap(starter: SlotAssignment, bench: SlotAssignment): boolean {
  const need = rolesOf(starter);
  const have = rolesOf(bench);
  if (need.size === 0 || have.size === 0) {
    // Por special-case by label
    if (starter.slotLabel === 'Por' || bench.slotLabel === 'Por') {
      return starter.slotLabel === 'Por' && (bench.slotLabel === 'Por' || have.has('Por'));
    }
    return false;
  }
  for (const r of have) {
    if (need.has(r)) return true;
  }
  // bench list uses slotLabel "bench" — rely on slotRoles only
  return false;
}

/**
 * Swap starter at index with a bench player (by playerId).
 * Returns a new lineup response with recalculated scoreTotale.
 */
export function swapStarterWithBench(
  lineup: LineupOptimizeResponse,
  starterPlayerId: string,
  benchPlayerId: string,
): LineupOptimizeResponse | null {
  const xi = [...(lineup.startingXi ?? [])];
  const bench = [...(lineup.bench ?? [])];
  const si = xi.findIndex((p) => p.playerId === starterPlayerId);
  const bi = bench.findIndex((p) => p.playerId === benchPlayerId);
  if (si < 0 || bi < 0) return null;

  const starter = xi[si];
  const sub = bench[bi];
  if (!canSwap(starter, sub)) return null;

  // Incoming keeps the slot metadata; outgoing goes to bench with their roles
  const newStarter: SlotAssignment = {
    ...sub,
    slotLabel: starter.slotLabel,
    slotRoles: [...starter.slotRoles],
  };
  const newBench: SlotAssignment = {
    ...starter,
    slotLabel: 'bench',
    slotRoles: [...starter.slotRoles],
  };

  xi[si] = newStarter;
  bench[bi] = newBench;

  const scoreTotale = xi.reduce((acc, p) => acc + (p.expectedScore || 0), 0);

  return {
    ...lineup,
    startingXi: xi,
    bench,
    scoreTotale: Math.round(scoreTotale * 1000) / 1000,
    notes: [
      ...(lineup.notes ?? []).filter((n) => !n.startsWith('Sostituzione manuale:')),
      `Sostituzione manuale: ${starter.playerName} ↔ ${sub.playerName}`,
    ],
  };
}
