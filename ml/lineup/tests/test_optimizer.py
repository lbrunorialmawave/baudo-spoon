"""Unit tests for the exact lineup optimizer."""

from __future__ import annotations

from ml.lineup.optimizer import (
    LineupCandidate,
    OptimizeResult,
    compute_ev,
    opponent_adjustment,
    optimize_lineup,
)
from ml.optimizer.formations import MANTRA_FORMATIONS


def _c(
    pid: str,
    name: str,
    roles: set[str],
    ev: float,
    sp: float = 0.8,
) -> LineupCandidate:
    return LineupCandidate(
        player_id=pid,
        name=name,
        eligible_roles=frozenset(roles),
        expected_value=ev,
        starter_probability=sp,
    )


def _balanced_squad() -> list[LineupCandidate]:
    """Enough players to cover a 4-3-3 / 3-5-2 style module."""
    return [
        _c("gk1", "Maignan", {"Por"}, 6.5, 0.95),
        _c("gk2", "Backup", {"Por"}, 5.0, 0.1),
        # defence
        _c("dd1", "Di Lorenzo", {"Dd", "B"}, 6.2, 0.9),
        _c("dc1", "Bremer", {"Dc"}, 6.8, 0.95),
        _c("dc2", "Gabbia", {"Dc"}, 5.8, 0.7),
        _c("dc3", "Bastoni", {"Dc", "B"}, 6.5, 0.85),
        _c("ds1", "Dimarco", {"Ds", "E"}, 7.0, 0.9),
        # mid
        _c("e1", "Spinazzola", {"E", "Ds"}, 5.5, 0.6),
        _c("m1", "Barella", {"C", "M"}, 7.2, 0.95),
        _c("m2", "Calhanoglu", {"C", "T"}, 7.0, 0.9),
        _c("m3", "Locatelli", {"M", "C"}, 6.0, 0.75),
        _c("t1", "Zielinski", {"T", "C"}, 6.3, 0.7),
        # attack
        _c("w1", "Leao", {"W", "A"}, 7.5, 0.85),
        _c("w2", "Politano", {"W", "A"}, 6.4, 0.7),
        _c("a1", "Thuram", {"A"}, 7.1, 0.9),
        _c("a2", "Hojlund", {"A", "Pc"}, 6.6, 0.8),
        _c("pc1", "Scamacca", {"Pc", "A"}, 6.0, 0.5),
    ]


def test_optimize_returns_feasible_choice():
    result = optimize_lineup(_balanced_squad())
    assert isinstance(result, OptimizeResult)
    assert result.chosen is not None
    assert result.chosen.feasible
    assert result.chosen.gk is not None
    assert result.chosen.gk.player_name == "Maignan"
    assert result.chosen.score_totale > 0
    # 1 GK + 10 outfield
    assert len(result.chosen.assignments) == 10
    assert result.chosen.formation in {f.label for f in MANTRA_FORMATIONS}


def test_bench_excludes_starters():
    result = optimize_lineup(_balanced_squad())
    assert result.chosen is not None
    used = {a.player_id for a in result.chosen.assignments}
    used.add(result.chosen.gk.player_id)
    for b in result.bench:
        assert b.player_id not in used


def test_low_starter_prob_excluded():
    squad = _balanced_squad()
    # force only backup GK with high SP → primary still chosen due to higher EV
    # kill all outfield high SP for W/A to make 3-4-3 hard — instead:
    # set one key player below threshold and ensure he is not starting
    low = _c("ghost", "Ghost W", {"W", "A"}, 99.0, sp=0.05)
    squad = squad + [low]
    result = optimize_lineup(squad, min_starter_prob=0.15)
    assert result.chosen is not None
    used_names = {a.player_name for a in result.chosen.assignments}
    assert "Ghost W" not in used_names


def test_no_gk_infeasible():
    squad = [c for c in _balanced_squad() if "Por" not in c.eligible_roles]
    result = optimize_lineup(squad)
    assert result.chosen is None
    assert any(not r.feasible for r in result.alternatives)


def test_subset_formations():
    result = optimize_lineup(_balanced_squad(), formations=["4-3-3", "3-5-2"])
    assert result.chosen is not None
    assert result.chosen.formation in ("4-3-3", "3-5-2")
    alt_labels = {a.formation for a in result.alternatives}
    assert alt_labels <= {"4-3-3", "3-5-2"}


def test_empty_squad():
    result = optimize_lineup([])
    assert result.chosen is None
    assert result.bench == ()


def test_compute_ev_and_adjustment():
    assert compute_ev(fp_ibrido_voto=7.0, starter_probability=0.8, opponent_adjustment=1.1) == 7.0 * 0.8 * 1.1
    # weak opponent → attack boost
    adj = opponent_adjustment("A", 0.2)
    assert adj > 1.0
    # strong opponent → attack penalty
    adj2 = opponent_adjustment("A", 0.8)
    assert adj2 < 1.0
    # defence milder
    adj_d = opponent_adjustment("Dc", 0.2)
    assert 0.90 <= adj_d <= 1.10


def test_optimality_vs_manual_best_single_formation():
    """On a tiny pool, Hungarian must match brute-force best assignment."""
    # 2 slots Dc, 2 players — only one feasible pairing order by EV
    players = [
        _c("a", "A", {"Dc"}, 8.0),
        _c("b", "B", {"Dc"}, 6.0),
        _c("gk", "GK", {"Por"}, 5.0),
    ]
    # custom formation with 2 Dc only — use  a real one that needs more;
    # instead verify total score includes both when formation needs 2 Dc
    from ml.optimizer.formations import MantraFormation, SlotRequirement

    mini = MantraFormation(
        label="test-2dc",
        slots=(SlotRequirement(roles=frozenset({"Dc"}), count=2, label="Dc"),),
    )
    result = optimize_lineup(players, formations=[mini])
    assert result.chosen is not None
    assert result.chosen.feasible
    # both Dc selected, score = 8+6 + gk
    assert abs(result.chosen.score_totale - (8.0 + 6.0 + 5.0)) < 1e-6
