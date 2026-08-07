"""Tests for Fase 1 — MANTRA ruleset support in ``AuctionConfig``.

Scope: domain-only (``ml/auction/models.py``). Verifies that:

* ``ruleset="CLASSIC"`` (default) is 100% backward compatible with existing
  callers — same validation, same error messages, same default quotas.
* ``ruleset="MANTRA"`` accepts the 12 Mantra role quotas without raising,
  either via the shared ``MANTRA_DEFAULT_QUOTAS`` convenience default or an
  explicit custom mapping.
* Cross-ruleset misuse (CLASSIC keys under MANTRA and vice versa) is
  rejected with a clear error.

See plan §5.1 Fase 1 for the acceptance criteria this file covers.
"""

from __future__ import annotations

import pytest

from ml.auction.models import AuctionConfig
from ml.mantra.roles import ALL_ROLES as MANTRA_ALL_ROLES
from ml.optimizer.models import MANTRA_DEFAULT_QUOTAS

# ---------------------------------------------------------------------------
# CLASSIC — unchanged behaviour (G2: zero regressions)
# ---------------------------------------------------------------------------


def test_classic_is_default_ruleset() -> None:
    cfg = AuctionConfig(num_participants=8)
    assert cfg.ruleset == "CLASSIC"
    assert cfg.role_quotas == {"P": 3, "D": 8, "C": 8, "A": 6}


def test_classic_explicit_ruleset_matches_default() -> None:
    cfg = AuctionConfig(num_participants=8, ruleset="CLASSIC")
    assert cfg.role_quotas == {"P": 3, "D": 8, "C": 8, "A": 6}


def test_classic_rejects_missing_role() -> None:
    with pytest.raises(ValueError, match="P/D/C/A"):
        AuctionConfig(
            num_participants=8,
            role_quotas={"P": 3, "D": 8, "C": 8},
        )


def test_classic_rejects_mantra_roles() -> None:
    with pytest.raises(ValueError, match="P/D/C/A"):
        AuctionConfig(
            num_participants=8,
            ruleset="CLASSIC",
            role_quotas=dict(MANTRA_DEFAULT_QUOTAS),
        )


def test_classic_rejects_non_positive_quota() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        AuctionConfig(
            num_participants=8,
            role_quotas={"P": 3, "D": 8, "C": 8, "A": 0},
        )


# ---------------------------------------------------------------------------
# MANTRA — new behaviour (G1)
# ---------------------------------------------------------------------------


def test_mantra_falls_back_to_default_quotas_when_unset() -> None:
    """Caller opts into MANTRA without overriding role_quotas."""
    cfg = AuctionConfig(num_participants=8, ruleset="MANTRA")
    assert cfg.role_quotas == MANTRA_DEFAULT_QUOTAS
    assert set(cfg.role_quotas.keys()) == set(MANTRA_ALL_ROLES)


def test_mantra_accepts_explicit_default_quotas() -> None:
    cfg = AuctionConfig(
        num_participants=8,
        ruleset="MANTRA",
        role_quotas=dict(MANTRA_DEFAULT_QUOTAS),
    )
    assert cfg.role_quotas == MANTRA_DEFAULT_QUOTAS


def test_mantra_accepts_custom_subset_quotas() -> None:
    """A custom quota mapping using only a subset of valid Mantra roles."""
    custom = {"Por": 3, "Dc": 4, "C": 8, "A": 4, "Pc": 2, "T": 2, "W": 2}
    cfg = AuctionConfig(
        num_participants=8,
        ruleset="MANTRA",
        role_quotas=custom,
    )
    assert cfg.role_quotas == custom


def test_mantra_rejects_unknown_role() -> None:
    bad = dict(MANTRA_DEFAULT_QUOTAS)
    bad["ZZ"] = 1
    with pytest.raises(ValueError, match="ZZ"):
        AuctionConfig(num_participants=8, ruleset="MANTRA", role_quotas=bad)


def test_mantra_rejects_non_positive_quota() -> None:
    bad = dict(MANTRA_DEFAULT_QUOTAS)
    bad["Por"] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        AuctionConfig(num_participants=8, ruleset="MANTRA", role_quotas=bad)


def test_mantra_rejects_empty_quotas() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AuctionConfig(num_participants=8, ruleset="MANTRA", role_quotas={})


# ---------------------------------------------------------------------------
# Ruleset field itself
# ---------------------------------------------------------------------------


def test_invalid_ruleset_rejected() -> None:
    with pytest.raises(ValueError, match="CLASSIC or MANTRA"):
        AuctionConfig(num_participants=8, ruleset="EXOTIC")  # type: ignore[arg-type]


def test_mantra_default_quotas_sum_matches_classic_squad_size() -> None:
    """Same total squad size (25) across rulesets, for parity with Optimizer."""
    assert sum(MANTRA_DEFAULT_QUOTAS.values()) == sum(
        {"P": 3, "D": 8, "C": 8, "A": 6}.values()
    )
