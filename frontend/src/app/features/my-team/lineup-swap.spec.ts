import { describe, expect, it } from 'vitest';
import { canSwap, rolesOf, swapStarterWithBench } from './lineup-swap';
import { LineupOptimizeResponse, SlotAssignment } from '../../core/models/my-team.models';

function slot(
  partial: Partial<SlotAssignment> & Pick<SlotAssignment, 'playerId' | 'playerName' | 'slotLabel'>,
): SlotAssignment {
  return {
    slotRoles: [],
    expectedScore: 6,
    starterProbability: 0.8,
    ...partial,
  };
}

describe('lineup-swap', () => {
  it('rolesOf splits label when slotRoles empty', () => {
    const r = rolesOf(slot({ playerId: '1', playerName: 'A', slotLabel: 'W/A#1' }));
    expect(r.has('W')).toBe(true);
    expect(r.has('A')).toBe(true);
  });

  it('canSwap requires role intersection', () => {
    const starter = slot({
      playerId: 's',
      playerName: 'Starter',
      slotLabel: 'W/A',
      slotRoles: ['W', 'A'],
    });
    const ok = slot({
      playerId: 'b',
      playerName: 'Wing',
      slotLabel: 'bench',
      slotRoles: ['W'],
    });
    const no = slot({
      playerId: 'c',
      playerName: 'Mid',
      slotLabel: 'bench',
      slotRoles: ['C'],
    });
    expect(canSwap(starter, ok)).toBe(true);
    expect(canSwap(starter, no)).toBe(false);
  });

  it('swapStarterWithBench updates XI and score', () => {
    const lineup: LineupOptimizeResponse = {
      contextId: 'x',
      teamName: 'T',
      sheetName: 'Divisione B',
      chosenFormation: '4-3-3',
      scoreTotale: 12,
      startingXi: [
        slot({
          playerId: '1',
          playerName: 'Leao',
          slotLabel: 'W/A',
          slotRoles: ['W', 'A'],
          expectedScore: 7,
        }),
        slot({
          playerId: '2',
          playerName: 'Barella',
          slotLabel: 'C',
          slotRoles: ['C'],
          expectedScore: 5,
        }),
      ],
      bench: [
        slot({
          playerId: '3',
          playerName: 'Politano',
          slotLabel: 'bench',
          slotRoles: ['W', 'A'],
          expectedScore: 6.5,
        }),
      ],
      alternativesConsidered: [],
      notes: [],
    };

    const next = swapStarterWithBench(lineup, '1', '3');
    expect(next).not.toBeNull();
    expect(next!.startingXi.find((p) => p.slotLabel === 'W/A')?.playerName).toBe('Politano');
    expect(next!.bench.some((p) => p.playerName === 'Leao')).toBe(true);
    expect(next!.scoreTotale).toBeCloseTo(6.5 + 5, 5);
  });

  it('rejects incompatible swap', () => {
    const lineup: LineupOptimizeResponse = {
      contextId: 'x',
      teamName: 'T',
      sheetName: 'B',
      startingXi: [
        slot({
          playerId: '1',
          playerName: 'GK',
          slotLabel: 'Por',
          slotRoles: ['Por'],
          expectedScore: 6,
        }),
      ],
      bench: [
        slot({
          playerId: '2',
          playerName: 'Att',
          slotLabel: 'bench',
          slotRoles: ['A'],
          expectedScore: 7,
        }),
      ],
      alternativesConsidered: [],
      notes: [],
    };
    expect(swapStarterWithBench(lineup, '1', '2')).toBeNull();
  });
});
