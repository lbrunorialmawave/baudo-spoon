"""PR6 — shadow mode, conservation under shadow, rollout helpers."""

from __future__ import annotations

from scraper.src.player_career_scraper import ForeignStatsResult


def test_shadow_result_conservation_ok():
    r = ForeignStatsResult(
        candidates=10,
        fetched=8,
        unresolved=2,
        would_persist=7,
        would_skip=0,
        skipped_invalid=1,
        skipped_other=0,
        shadow=True,
    )
    r.assert_conservation()
    assert r.invariant_ok is True
    assert r.persistence_rate == 7 / 8


def test_shadow_result_to_dict_includes_would_fields():
    r = ForeignStatsResult(
        candidates=5,
        fetched=5,
        would_persist=5,
        shadow=True,
    )
    r.assert_conservation()
    d = r.to_dict()
    assert d["shadow"] is True
    assert d["would_persist"] == 5
    assert "would_skip" in d
    assert d["persistence_rate"] == 100.0


def test_non_shadow_to_dict_omits_would_fields():
    r = ForeignStatsResult(
        candidates=3, fetched=3, persisted=3, unresolved=0, shadow=False
    )
    r.assert_conservation()
    d = r.to_dict()
    assert d["shadow"] is False
    assert "would_persist" not in d


def test_shadow_conservation_detects_mismatch():
    r = ForeignStatsResult(
        candidates=10,
        fetched=8,
        unresolved=2,
        would_persist=3,  # 3+0+0+0 != 8
        would_skip=0,
        skipped_invalid=0,
        skipped_other=0,
        shadow=True,
    )
    r.assert_conservation()
    assert r.invariant_ok is False


def test_empty_candidates_ok():
    r = ForeignStatsResult(candidates=0, shadow=True)
    r.assert_conservation()
    assert r.invariant_ok is True
    assert r.persistence_rate is None


def test_health_checks_structure_without_db():
    """_health_checks is defined and documents the expected gate names."""
    from pathlib import Path

    mod_path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_foreign_stats.py"
    if not mod_path.exists():
        mod_path = Path("scripts/backfill_foreign_stats.py")
    src = mod_path.read_text()
    assert "def _health_checks" in src
    for name in (
        "latest_view_readable",
        "target_aware_view_readable",
        "lineage_columns_queryable",
        "foreign_sentinel_queryable",
        "conservation_invariants",
        "candidates_accounted",
        "season_resolution_ran",
    ):
        assert name in src
    assert "--health" in src
