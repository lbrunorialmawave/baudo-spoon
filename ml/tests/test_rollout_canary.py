"""Tests for ``ml.rollout.canary`` (WS14, plan §16.1).

The canary report is the gate artefact for the SHADOW → ACTIVE
promotion: ``anomalies.remaining`` must be 0 in a healthy build, and
the report must carry a canonical ``config_hash`` (WS16) so the
promotion-gate checker can verify the artefact matches the active
configuration.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import pytest

# All tests in this module need a database URL for pydantic-settings.
# We do not connect to a real DB here — the MLConfig validator only
# needs the URL to be non-empty.
os.environ.setdefault(
    "ML_DATABASE_URL",
    "postgresql+psycopg2://fake:fake@localhost:5432/fake",
)

from ml.config import MLConfig  # noqa: E402
from ml.rollout import build_canary_report  # noqa: E402
from ml.rollout.canary import (  # noqa: E402
    CANARY_REPORT_VERSION,
    _classify_anomalies,
    _effective_fantavoto,
    _per_role_standard_median,
)
from ml.rollout.config_hash import verify_config_hash  # noqa: E402
from ml.tests.fixtures.limited_cohort_canary import (  # noqa: E402
    CANARY_ANOMALY_IDS,
    build_limited_cohort_canary,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> MLConfig:
    return MLConfig()


@pytest.fixture
def fixture_df() -> Any:
    return build_limited_cohort_canary()


# ── Schema invariants ───────────────────────────────────────────────────────


class TestReportSchema:
    def test_required_keys_present(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        for key in (
            "anomalies",
            "canary_anomalies_remaining",
            "config_hash",
            "config",
        ):
            assert key in report, f"missing key {key}"

    def test_anomalies_block_shape(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        anomalies = report["anomalies"]
        for key in ("total", "resolved", "remaining", "remaining_count", "details"):
            assert key in anomalies, f"missing anomalies.{key}"
        # Aliases must agree.
        assert anomalies["remaining"] == anomalies["remaining_count"]
        assert (
            anomalies["remaining"]
            == report["canary_anomalies_remaining"]
        )

    def test_config_hash_format(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        assert report["config_hash"].startswith("sha256:")
        # Hash must verify against the embedded config block.
        assert verify_config_hash(report["config"], report["config_hash"])

    def test_version_stamped(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        assert report["version"] == CANARY_REPORT_VERSION

    def test_known_anomaly_ids_recorded(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        assert sorted(CANARY_ANOMALY_IDS) == report["known_anomaly_ids"]


# ── Behaviour on the default fixture ────────────────────────────────────────


class TestHealthyFixture:
    """The synthetic fixture is designed to be *clean* under the
    safety net — Adzic's raw 8.20 must drop to <= the FWD STANDARD
    median, leaving 0 remaining anomalies.  This is the regression
    the workflow's Phase-6 validator relies on."""

    def test_default_fixture_is_clean(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        assert report["anomalies"]["remaining"] == 0
        assert report["gate_passed"] is True
        assert report["anomalies"]["total"] >= 1  # at least Adzic

    def test_adzic_resolved_below_role_median(self, cfg: MLConfig) -> None:
        report = build_canary_report(cfg)
        adzic = next(
            d for d in report["anomalies"]["details"] if d["player_id"] == "adzic-163"
        )
        assert adzic["resolved"] is True
        assert adzic["role"] == "FWD"
        assert adzic["effective_fantavoto"] <= adzic["role_standard_median"]


# ── Internal helpers ────────────────────────────────────────────────────────


class TestEffectiveFantavoto:
    def test_standard_rows_keep_their_value(self, cfg: MLConfig, fixture_df: Any) -> None:
        out = _effective_fantavoto(
            fixture_df,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
        )
        std = out[out["sample_cohort"] == "STANDARD"]
        # Reliability weight for STANDARD minutes must be ~1.0, so the
        # effective value should match the raw prediction to within
        # float precision.
        for _, row in std.iterrows():
            assert row["effective_fantavoto"] == pytest.approx(
                row["predicted_fantavoto"], rel=1e-6
            )
            assert row["reliability_weight"] == pytest.approx(1.0, abs=1e-9)

    def test_limited_rows_are_discounted(self, cfg: MLConfig, fixture_df: Any) -> None:
        out = _effective_fantavoto(
            fixture_df,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
        )
        limited = out[out["sample_cohort"] == "LIMITED"]
        for _, row in limited.iterrows():
            assert 0.0 < row["reliability_weight"] < 1.0
            assert row["effective_fantavoto"] < row["predicted_fantavoto"]

    def test_sample_weights_computed(self, cfg: MLConfig, fixture_df: Any) -> None:
        out = _effective_fantavoto(
            fixture_df,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
        )
        assert "sample_weight" in out.columns
        # All weights must be in [0, 1] (the function's invariant).
        assert ((out["sample_weight"] >= 0) & (out["sample_weight"] <= 1)).all()
        # INSUFFICIENT rows (mins_played < min_minutes_hard) are
        # *designed* to receive 0.0; check the LIMITED+STANDARD rows
        # have strictly positive weights.
        eligible = out[out["mins_played"] >= cfg.min_minutes_hard]
        assert (eligible["sample_weight"] > 0).all()
        # Monotone in minutes: heavier minutes → larger (or equal) weight.
        ordered = out.sort_values("mins_played")
        weights = ordered["sample_weight"].tolist()
        assert weights == sorted(weights)


class TestPerRoleStandardMedian:
    def test_only_standard_rows_counted(self, fixture_df: Any) -> None:
        medians = _per_role_standard_median(fixture_df)
        # Fixture has STANDARD rows for FWD, MID, DEF, GK.
        for role in ("FWD", "MID", "DEF", "GK"):
            assert role in medians, f"missing role {role}"
        # LIMITED/INSUFFICIENT should not appear.
        assert "LIMITED" not in medians

    def test_median_matches_input(
        self, cfg: MLConfig, fixture_df: Any
    ) -> None:
        evaluated = _effective_fantavoto(
            fixture_df,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
        )
        medians = _per_role_standard_median(evaluated)
        std = evaluated[evaluated["sample_cohort"] == "STANDARD"]
        for role, expected in medians.items():
            rows = std[std["canonical_role"] == role]
            assert expected == pytest.approx(rows["predicted_fantavoto"].median())


class TestClassifyAnomalies:
    def test_known_id_marked_resolved_when_below_threshold(
        self, cfg: MLConfig, fixture_df: Any
    ) -> None:
        evaluated = _effective_fantavoto(
            fixture_df,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
        )
        medians = _per_role_standard_median(evaluated)
        findings = _classify_anomalies(
            evaluated, standard_medians=medians, known_anomaly_ids=CANARY_ANOMALY_IDS
        )
        assert findings, "no findings for known anomalies"
        for f in findings:
            assert f["resolved"] is True

    def test_no_standard_baseline_fails_closed(
        self, cfg: MLConfig, fixture_df: Any
    ) -> None:
        # Drop every STANDARD row and re-classify: no baseline → all
        # known anomalies must be unresolved so the gate fails closed.
        trimmed = fixture_df[fixture_df["sample_cohort"] != "STANDARD"].reset_index(
            drop=True
        )
        evaluated = _effective_fantavoto(
            trimmed,
            min_minutes_hard=cfg.min_minutes_hard,
            standard_minutes=cfg.min_minutes,
        )
        medians = _per_role_standard_median(evaluated)
        findings = _classify_anomalies(
            evaluated,
            standard_medians=medians,
            known_anomaly_ids=CANARY_ANOMALY_IDS,
        )
        assert all(f["resolved"] is False for f in findings)


# ── Determinism / hash stability ────────────────────────────────────────────


class TestDeterminism:
    def test_same_config_produces_same_hash(self, cfg: MLConfig) -> None:
        a = build_canary_report(cfg)
        b = build_canary_report(cfg)
        assert a["config_hash"] == b["config_hash"]

    def test_different_config_produces_different_hash(self, cfg: MLConfig) -> None:
        a = build_canary_report(cfg)
        other = copy.copy(cfg)
        # Mutate a flag that is part of the config snapshot.
        object.__setattr__(other, "weighting_strategy", "linear")
        b = build_canary_report(other)
        assert a["config_hash"] != b["config_hash"]


# ── Custom override paths ───────────────────────────────────────────────────


class TestOverrides:
    def test_custom_known_ids(self, cfg: MLConfig, fixture_df: Any) -> None:
        # Force the safety net off by overriding the known-anomaly set
        # to an ID that is NOT in the fixture → total = 0.
        report = build_canary_report(
            cfg, df=fixture_df, known_anomaly_ids=frozenset({"ghost-id"})
        )
        assert report["anomalies"]["total"] == 0
        assert report["anomalies"]["remaining"] == 0
        assert report["gate_passed"] is True
