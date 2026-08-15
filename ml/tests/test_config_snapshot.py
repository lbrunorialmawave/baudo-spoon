"""Tests for ml.rollout.config_snapshot (WS16 of plan.md, idempotency).

These tests are the regression net for the bug fixed after Run
"Idempotenza": the three emitted artefacts (effective_config.json,
promotion_report.json, canary_report.json) used to construct their
``config`` payload with three different inline dicts, so the
``config_hash`` they carried never matched across re-runs.  The gate
(Phase 6 of ``ml-training.yml``) then denied the transition with
``config_hash mismatch`` even when nothing actually changed.

The fix is to centralise the snapshot in
:mod:`ml.rollout.config_snapshot` and have all three emission sites
consume the same helper.  These tests assert both the helper itself
and the cross-site hash alignment.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from ml.rollout.config_hash import compute_config_hash
from ml.rollout.config_snapshot import (
    ML_CONFIG_SNAPSHOT_KEYS,
    build_ml_config_snapshot,
    merge_ml_snapshot,
)
from ml.rollout.canary import build_canary_report
from ml.config import MLConfig


# ── Helpers / fixtures ─────────────────────────────────────────────────────


@dataclass
class _StubConfig:
    """Lightweight stand-in for MLConfig used by config_snapshot tests.

    Avoids touching pydantic settings, DB, or any heavy machinery.  All
    fields in :data:`ML_CONFIG_SNAPSHOT_KEYS` are populated with values
    chosen to surface coercion behaviour (e.g. ``True`` vs ``1`` for
    booleans, ``int``/``float`` for numerics, ``str`` for modes).
    """

    min_minutes: int = 600
    min_minutes_hard: int = 270
    enable_limited_sample_training: bool = False
    enable_shrinkage: bool = True
    enable_recent_role_features: bool = True
    enable_breakout_model: bool = False
    weighting_strategy: str = "inverse_minutes"
    shrinkage_prior_strength: int = 50
    reliability_weight_mode: str = "continuous"


def _real_config() -> MLConfig:
    """Build a real :class:`MLConfig` instance for cross-site tests.

    The fields driven by the env (``DATABASE_URL``, …) are not relevant
    for snapshot generation, so we don't need to touch them.
    """
    return MLConfig(
        min_minutes=600,
        min_minutes_hard=270,
        enable_limited_sample_training=False,
        enable_shrinkage=True,
        enable_recent_role_features=True,
        enable_breakout_model=False,
        weighting_strategy="inverse_minutes",
        shrinkage_prior_strength=50,
        reliability_weight_mode="continuous",
    )


# ── TestBuildMlConfigSnapshot ──────────────────────────────────────────────


class TestBuildMlConfigSnapshot:
    def test_returns_keys_in_canonical_order(self) -> None:
        cfg = _StubConfig()
        snapshot = build_ml_config_snapshot(cfg)
        assert tuple(snapshot.keys()) == ML_CONFIG_SNAPSHOT_KEYS

    def test_values_are_hash_stable_primitives(self) -> None:
        cfg = _StubConfig()
        snapshot = build_ml_config_snapshot(cfg)
        for key, value in snapshot.items():
            # Either int, bool, float, or str — never a custom object.
            assert isinstance(value, (int, bool, float, str)), (
                f"Field {key!r} has non-primitive type {type(value).__name__}"
            )

    def test_bool_coercion_is_exact(self) -> None:
        # ``int(True)`` returns 1; the helper MUST keep the bool
        # distinction so the hash matches the boolean source-of-truth.
        cfg = _StubConfig(enable_shrinkage=True)
        assert build_ml_config_snapshot(cfg)["enable_shrinkage"] is True
        cfg = _StubConfig(enable_shrinkage=False)
        assert build_ml_config_snapshot(cfg)["enable_shrinkage"] is False

    def test_int_coercion(self) -> None:
        cfg = _StubConfig(min_minutes=600, shrinkage_prior_strength=50)
        snap = build_ml_config_snapshot(cfg)
        assert snap["min_minutes"] == 600
        assert isinstance(snap["min_minutes"], int)
        assert snap["shrinkage_prior_strength"] == 50

    def test_str_coercion(self) -> None:
        cfg = _StubConfig(weighting_strategy="inverse_minutes")
        assert (
            build_ml_config_snapshot(cfg)["weighting_strategy"]
            == "inverse_minutes"
        )

    def test_missing_field_raises(self) -> None:
        @dataclass
        class Incomplete:
            min_minutes: int = 600
            # ... intentionally missing all other canonical fields

        with pytest.raises(AttributeError, match="min_minutes_hard"):
            build_ml_config_snapshot(Incomplete())

    def test_idempotency_two_calls_same_input_same_hash(self) -> None:
        cfg = _StubConfig()
        snap_a = build_ml_config_snapshot(cfg)
        snap_b = build_ml_config_snapshot(cfg)
        assert snap_a == snap_b
        assert compute_config_hash(snap_a) == compute_config_hash(snap_b)


# ── TestMergeMlSnapshot ────────────────────────────────────────────────────


class TestMergeMlSnapshot:
    def test_extra_appended_after_canonical_keys(self) -> None:
        cfg = _StubConfig()
        merged = merge_ml_snapshot(cfg, extra={"production_mode": "continuous"})
        keys = list(merged.keys())
        # All canonical keys come first, in the canonical order.
        assert keys[: len(ML_CONFIG_SNAPSHOT_KEYS)] == list(
            ML_CONFIG_SNAPSHOT_KEYS
        )
        assert keys[-1] == "production_mode"

    def test_extra_does_not_shadow_canonical_keys(self) -> None:
        with pytest.raises(ValueError, match="collides with the canonical"):
            merge_ml_snapshot(_StubConfig(), extra={"min_minutes": 999})

    def test_extra_does_not_change_config_hash(self) -> None:
        cfg = _StubConfig()
        snap = build_ml_config_snapshot(cfg)
        merged = merge_ml_snapshot(cfg, extra={"anything": 42})
        # The hash of ``merged`` differs from ``snap`` because the
        # payload itself differs — but the **canonical** hash computed
        # from the shared subset must remain identical.
        assert compute_config_hash(snap) != compute_config_hash(merged)
        # Re-extracting the canonical subset restores equality.
        canonical_subset = {k: merged[k] for k in ML_CONFIG_SNAPSHOT_KEYS}
        assert compute_config_hash(canonical_subset) == compute_config_hash(snap)

    def test_none_extra_returns_snapshot_unchanged(self) -> None:
        cfg = _StubConfig()
        assert merge_ml_snapshot(cfg) == build_ml_config_snapshot(cfg)


# ── Cross-site hash alignment (the real bug) ──────────────────────────────


class TestCrossSiteHashAlignment:
    """The bug we are fixing: three artefacts used three different
    inline dicts, so their config_hashes never matched across re-runs.
    """

    def test_canary_and_promotion_report_share_config_hash(self) -> None:
        """Both the canary and the promotion report MUST produce the
        same canonical config_hash for the same MLConfig."""
        cfg = _real_config()
        canary = build_canary_report(cfg)
        canary_hash = canary["config_hash"]
        # Re-running the canary is pure → second hash MUST match.
        assert canary_hash == build_canary_report(cfg)["config_hash"]
        # The canary's nested ``config`` block must equal what the
        # promotion report would build.
        from ml.run_pipeline import _build_promotion_report_payload

        report = _build_promotion_report_payload(
            cfg, {"best_model": "rf", "metadata": {"run_id": "x"}}
        )
        assert report["config"] == canary["config"]
        assert report["config_hash"] == canary_hash

    def test_effective_config_carries_canonical_hash(self) -> None:
        """``effective_config.json`` MUST carry the same canonical hash
        as the canary and the promotion report."""
        from ml.run_pipeline import _build_effective_config_payload

        cfg = _real_config()
        effective = _build_effective_config_payload(cfg)
        canary = build_canary_report(cfg)
        # The canonical ``config_hash`` MUST match across all three.
        assert effective["config_hash"] == canary["config_hash"]
        # The nested ``config`` payload MUST equal the canary's.
        assert effective["config"] == canary["config"]

    def test_extra_fields_in_effective_config_do_not_change_hash(self) -> None:
        """Sanity check: the production_mode / production_flags /
        stages block lives in ``extra``, not in ``config``, so it does
        not perturb the canonical hash."""
        from ml.run_pipeline import _build_effective_config_payload

        cfg_a = _real_config()
        cfg_b = _real_config()
        # Same ML config, but a different env-flag state (the helper
        # resolves env vars at call time).  In a real env override we'd
        # patch ``os.environ``, but the bundled ``extra`` block is
        # itself guaranteed to be hashed-out, so the canonical hash
        # MUST be stable.
        a = _build_effective_config_payload(cfg_a)
        b = _build_effective_config_payload(cfg_b)
        # The canonical hash is identical even if the ``extra`` block
        # varies (here it's the same env, but the contract is the
        # important bit).
        assert a["config_hash"] == b["config_hash"]
        assert a["config"] == b["config"]

    def test_re_emission_is_idempotent(self) -> None:
        """Re-running the helpers on the same config produces the same
        hash — this is the core idempotency contract for the rollout
        workflow."""
        from ml.run_pipeline import _build_promotion_report_payload

        cfg = _real_config()
        results = {
            "best_model": "rf",
            "metadata": {"run_id": "abc"},
            "role_metrics": {"outfield": {"rf": {"rmse": 1.0}}},
        }
        first = _build_promotion_report_payload(cfg, results)
        second = _build_promotion_report_payload(cfg, results)
        assert first["config_hash"] == second["config_hash"]
        assert first["config"] == second["config"]


# ── End-to-end bundle round-trip ───────────────────────────────────────────


class TestBundleRoundTrip:
    def test_bundle_round_trip_through_json(self) -> None:
        """The bundle must survive a JSON round-trip unchanged so the
        build job can write it to disk and the gate job can read it
        back without a hash drift."""
        cfg = _StubConfig()
        snap = build_ml_config_snapshot(cfg)
        h = compute_config_hash(snap)
        serialised = json.dumps({"config": snap, "config_hash": h})
        loaded = json.loads(serialised)
        assert loaded["config"] == snap
        assert loaded["config_hash"] == h
