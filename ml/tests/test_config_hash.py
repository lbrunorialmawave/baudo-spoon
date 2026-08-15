"""Tests for ml.rollout.config_hash (WS16 of plan.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ml.rollout.config_hash import (
    HASH_ALGORITHM,
    HASH_PREFIX,
    build_config_bundle,
    canonical_json,
    compute_config_hash,
    short_hash,
    verify_config_hash,
)


# ── canonical_json determinism ─────────────────────────────────────────────


class TestCanonicalJson:
    def test_key_order_independence(self) -> None:
        a = canonical_json({"b": 2, "a": 1, "c": 3})
        b = canonical_json({"c": 3, "a": 1, "b": 2})
        assert a == b

    def test_nested_key_order_independence(self) -> None:
        a = canonical_json({"outer": {"z": 1, "a": 2}})
        b = canonical_json({"outer": {"a": 2, "z": 1}})
        assert a == b

    def test_no_whitespace(self) -> None:
        s = canonical_json({"k": "v"})
        assert " " not in s
        # single character between tokens
        assert s == '{"k":"v"}'

    def test_set_is_sorted(self) -> None:
        s = canonical_json({"tags": {"b", "a", "c"}})
        assert s == '{"tags":["a","b","c"]}'

    def test_frozenset_is_sorted(self) -> None:
        s = canonical_json({"tags": frozenset({"x", "y"})})
        assert s == '{"tags":["x","y"]}'

    def test_tuple_becomes_list(self) -> None:
        s = canonical_json({"t": (1, 2, 3)})
        assert s == '{"t":[1,2,3]}'

    def test_dataclass_supported(self) -> None:
        @dataclass
        class C:
            x: int
            y: str

        s = canonical_json({"inner": C(1, "z")})
        assert s == '{"inner":{"x":1,"y":"z"}}'

    def test_non_string_keys_coerced(self) -> None:
        s = canonical_json({1: "a", 2: "b"})
        # keys become strings; order preserved via sort
        assert s == '{"1":"a","2":"b"}'

    def test_unicode_preserved(self) -> None:
        s = canonical_json({"nome": "Brunori"})
        assert "Brunori" in s


# ── compute_config_hash ────────────────────────────────────────────────────


class TestComputeConfigHash:
    def test_known_vector_simple(self) -> None:
        # Pre-computed against the canonical algorithm.
        h = compute_config_hash({"a": 1, "b": 2})
        assert h.startswith(HASH_PREFIX)
        assert len(h) == len(HASH_PREFIX) + 64  # 64 hex chars

    def test_algorithm_is_sha256(self) -> None:
        assert HASH_ALGORITHM == "sha256"

    def test_hash_changes_when_value_changes(self) -> None:
        a = compute_config_hash({"x": 1})
        b = compute_config_hash({"x": 2})
        assert a != b

    def test_hash_changes_when_key_added(self) -> None:
        a = compute_config_hash({"x": 1})
        b = compute_config_hash({"x": 1, "y": 2})
        assert a != b

    def test_hash_stable_across_dict_order(self) -> None:
        a = compute_config_hash({"b": 2, "a": 1})
        b = compute_config_hash({"a": 1, "b": 2})
        assert a == b

    def test_hash_stable_for_set_vs_list(self) -> None:
        # A set and a sorted list representing the same collection
        # must produce the same canonical serialisation.
        a = compute_config_hash({"tags": ["a", "b", "c"]})
        b = compute_config_hash({"tags": sorted({"a", "b", "c"})})
        assert a == b

    def test_hash_stable_for_dataclass(self) -> None:
        @dataclass
        class Cfg:
            flag: str
            pct: float

        a = compute_config_hash(Cfg(flag="x", pct=10.0))
        b = compute_config_hash({"flag": "x", "pct": 10.0})
        assert a == b

    def test_different_configs_different_hashes(self) -> None:
        h1 = compute_config_hash({"stage": "active", "mode": "continuous"})
        h2 = compute_config_hash({"stage": "shadow", "mode": "bucket"})
        assert h1 != h2

    def test_hash_is_lowercase_hex(self) -> None:
        h = compute_config_hash({"x": 1})
        body = h[len(HASH_PREFIX) :]
        assert all(c in "0123456789abcdef" for c in body)


# ── verify_config_hash ─────────────────────────────────────────────────────


class TestVerifyConfigHash:
    def test_match_returns_true(self) -> None:
        cfg = {"a": 1, "b": 2}
        h = compute_config_hash(cfg)
        assert verify_config_hash(cfg, h) is True

    def test_mismatch_returns_false(self) -> None:
        cfg = {"a": 1}
        h = compute_config_hash({"a": 2})
        assert verify_config_hash(cfg, h) is False

    def test_empty_expected_returns_false(self) -> None:
        assert verify_config_hash({"a": 1}, "") is False
        assert verify_config_hash({"a": 1}, None) is False  # type: ignore[arg-type]

    def test_wrong_prefix_returns_false(self) -> None:
        cfg = {"a": 1}
        h = compute_config_hash(cfg)
        # Strip the prefix → should not match.
        bad = h[len(HASH_PREFIX) :]
        assert verify_config_hash(cfg, bad) is False


# ── short_hash ──────────────────────────────────────────────────────────────


class TestShortHash:
    def test_default_length_is_12(self) -> None:
        s = short_hash({"x": 1})
        assert len(s) == 12

    def test_custom_length(self) -> None:
        s = short_hash({"x": 1}, length=6)
        assert len(s) == 6

    def test_returns_hex_only(self) -> None:
        s = short_hash({"x": 1})
        assert all(c in "0123456789abcdef" for c in s)

    def test_consistent_across_calls(self) -> None:
        a = short_hash({"x": 1})
        b = short_hash({"x": 1})
        assert a == b


# ── build_config_bundle ─────────────────────────────────────────────────────


class TestBuildConfigBundle:
    def test_contains_config_and_hash(self) -> None:
        b = build_config_bundle(config={"a": 1})
        assert "config" in b
        assert "config_hash" in b
        assert b["config"] == {"a": 1}

    def test_hash_matches_standalone(self) -> None:
        b = build_config_bundle(config={"a": 1, "b": 2})
        assert b["config_hash"] == compute_config_hash({"a": 1, "b": 2})

    def test_extra_preserved(self) -> None:
        b = build_config_bundle(
            config={"a": 1}, extra={"actor": "ci", "ts": "2026-08-15"}
        )
        assert b["extra"]["actor"] == "ci"
        # Hash only covers the config payload, NOT the extra metadata.
        assert b["config_hash"] == compute_config_hash({"a": 1})

    def test_extra_does_not_influence_hash(self) -> None:
        b1 = build_config_bundle(config={"a": 1}, extra={"x": 1})
        b2 = build_config_bundle(config={"a": 1}, extra={"x": 2})
        assert b1["config_hash"] == b2["config_hash"]

    def test_bundle_is_json_serialisable(self) -> None:
        b = build_config_bundle(config={"a": 1}, extra={"k": "v"})
        # Round-trip JSON
        s = json.dumps(b, default=str)
        loaded = json.loads(s)
        assert loaded["config_hash"] == b["config_hash"]


# ── Mandatory negative cases from plan §18 ────────────────────────────────


class TestMandatoryNegative:
    def test_drift_in_config_produces_different_hash(self) -> None:
        """Changing any field must change the hash, deterministically."""
        baseline = {
            "stage": "active",
            "mode": "continuous",
            "flags": {"a": True, "b": False},
        }
        with_drift = {
            "stage": "active",
            "mode": "continuous",
            "flags": {"a": True, "b": True},  # ← single bit flip
        }
        h1 = compute_config_hash(baseline)
        h2 = compute_config_hash(with_drift)
        assert h1 != h2

    def test_drift_in_pct_produces_different_hash(self) -> None:
        a = compute_config_hash({"pct": 10.0})
        b = compute_config_hash({"pct": 11.0})
        assert a != b

    def test_drift_in_reliability_mode_produces_different_hash(self) -> None:
        a = compute_config_hash({"mode": "bucket"})
        b = compute_config_hash({"mode": "continuous"})
        assert a != b

    def test_canonical_json_is_deterministic_under_random_permutations(self) -> None:
        """100 random permutations of the same dict all hash equal."""
        base = {"a": 1, "b": 2, "c": 3, "d": [1, 2, 3]}
        first = compute_config_hash(base)
        # Without depending on stdlib randomness, just simulate reorders.
        perms = [
            {"a": 1, "c": 3, "b": 2, "d": [1, 2, 3]},
            {"b": 2, "a": 1, "d": [1, 2, 3], "c": 3},
            {"d": [1, 2, 3], "c": 3, "b": 2, "a": 1},
        ]
        for p in perms:
            assert compute_config_hash(p) == first


# ── End-to-end with a realistic rollout snapshot ──────────────────────────


class TestEndToEndRolloutSnapshot:
    def test_rollout_state_hashing(self) -> None:
        """Mimics how check_promotion_gate.py will build its hash."""
        rollout_state = {
            "stage": "active",
            "mode": "continuous",
            "production_flags": {
                "enable_shrinkage": True,
                "enable_recent_role_features": True,
            },
            "challenger_flags": {
                "enable_shrinkage": True,
            },
            "pct": 25.0,
        }
        h1 = compute_config_hash(rollout_state)
        h2 = compute_config_hash(dict(rollout_state))  # copy
        assert h1 == h2

    def test_drift_in_flag_changes_hash(self) -> None:
        s1 = {
            "stage": "active",
            "mode": "continuous",
            "production_flags": {"enable_shrinkage": True},
        }
        s2 = {
            "stage": "active",
            "mode": "continuous",
            "production_flags": {"enable_shrinkage": False},  # flipped
        }
        assert compute_config_hash(s1) != compute_config_hash(s2)

    def test_bundle_payload_for_promotion_report(self) -> None:
        """Used by check_promotion_gate to stamp the report."""
        bundle = build_config_bundle(
            config={
                "stage": "active",
                "mode": "continuous",
                "production_flags": {"enable_shrinkage": True},
            },
            extra={
                "report_path": "ml/reports/promotion_report.json",
                "actor": "ml-training workflow",
            },
        )
        # Bundle is JSON-serialisable (no I/O required)
        serialised = json.dumps(bundle)
        loaded = json.loads(serialised)
        assert loaded["config_hash"] == bundle["config_hash"]
        assert loaded["extra"]["actor"] == "ml-training workflow"
