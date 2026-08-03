"""Tests for ml.mantra.roles.calcola_pool_esteso pool-fusion gating."""

from __future__ import annotations

from ml.mantra.roles import calcola_pool_esteso


def test_no_role_counts_keeps_legacy_always_fuse_behaviour() -> None:
    """Without role_counts, fusion happens unconditionally (back-compat)."""
    assert calcola_pool_esteso("Dc") == {"Dc", "B", "Dd", "Ds"}
    assert calcola_pool_esteso("E") == {"E", "M"}


def test_large_role_pool_does_not_fuse() -> None:
    """A role with a sample size >= soglia stands on its own, even when it
    appears as someone else's fusion target (e.g. Dc is a target of B/Dd/Ds)."""
    role_counts = {"Dc": 60, "B": 5, "Dd": 8, "Ds": 6}
    assert calcola_pool_esteso("Dc", role_counts, soglia=20) == {"Dc"}


def test_small_role_pool_still_fuses_with_role_counts() -> None:
    """A role with too few players still merges with its fusion partners."""
    role_counts = {"Dc": 60, "B": 5, "Dd": 8, "Ds": 6}
    assert calcola_pool_esteso("B", role_counts, soglia=20) == {"B", "Dc", "Dd", "Ds"}


def test_role_missing_from_counts_is_treated_as_zero_and_fuses() -> None:
    role_counts = {"Dc": 60}
    assert calcola_pool_esteso("B", role_counts, soglia=20) == {"B", "Dc", "Dd", "Ds"}


def test_symmetric_fusion_pairs_respect_their_own_counts() -> None:
    # E/M, T/W, A/Pc are mutual fusion pairs: each only fuses if it is
    # itself below soglia, independent of the partner's count.
    role_counts = {"E": 25, "M": 5, "T": 5, "W": 25, "A": 25, "Pc": 5}
    assert calcola_pool_esteso("E", role_counts, soglia=20) == {"E"}
    assert calcola_pool_esteso("M", role_counts, soglia=20) == {"M", "E"}
    assert calcola_pool_esteso("T", role_counts, soglia=20) == {"T", "W"}
    assert calcola_pool_esteso("W", role_counts, soglia=20) == {"W"}
    assert calcola_pool_esteso("A", role_counts, soglia=20) == {"A"}
    assert calcola_pool_esteso("Pc", role_counts, soglia=20) == {"Pc", "A"}


def test_role_without_fusion_entry_returns_itself() -> None:
    role_counts = {"Por": 3}
    assert calcola_pool_esteso("Por", role_counts, soglia=20) == {"Por"}


def test_role_at_exact_threshold_does_not_fuse() -> None:
    role_counts = {"Dc": 20, "B": 5, "Dd": 5, "Ds": 5}
    assert calcola_pool_esteso("Dc", role_counts, soglia=20) == {"Dc"}
