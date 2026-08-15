#!/usr/bin/env python3
"""Operational promotion gate for low-sample feature flags (WS5).

Reads an experiment harness report.json and exits non-zero when
cohort-aware quality gates fail. Intended to be run manually or in CI
before RolloutController.promote(new_stage=ACTIVE).

Exit codes:
    0 — gates passed
    1 — gates failed
    2 — invalid / missing report or I/O error
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Late import to keep the CLI fast on cold-start.
from ml.rollout.config_hash import compute_config_hash, verify_config_hash


DEFAULT_VARIANT = "C_shrinkage"
CONTROL_VARIANT = "A_control"

REQUIRED_REPORT_KEYS = (
    "mae",
    "rmse",
    "mae_by_cohort",
    "rmse_by_cohort",
    "phenom_leakage_rate",
    "canary_anomalies_remaining",
)


# ── Structured outcome (WS6) ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PromotionGateReport:
    """Structured outcome of a promotion-gate evaluation.

    The dataclass is the canonical result type consumed by
    :meth:`ml.rollout.RolloutController.promote_to_active` and by
    CI/logging code.  Field semantics:

    * ``passed`` — ``True`` when *no* failure was detected and the
      report was readable.
    * ``failures`` — tuple of human-readable failure strings.  Empty
      when ``passed`` is True.
    * ``report_path`` — str(path) of the report that was evaluated.
    * ``variant`` / ``control`` — the experiment variants used.
    * ``config_hash`` — canonical SHA-256 of the candidate config
      snapshot, when one was provided.
    * ``config_hash_status`` — ``"match"`` when the snapshot matches
      the report, ``"mismatch"`` when it doesn't, ``"skipped"`` when
      no snapshot was supplied, ``"report_missing_config_hash"``
      when the report didn't carry one (legacy / soft).
    """

    passed: bool
    failures: tuple[str, ...]
    report_path: str
    variant: str
    control: str = CONTROL_VARIANT
    config_hash: str | None = None
    config_hash_status: str | None = None
    config_snapshot_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failures"] = list(self.failures)
        return d


class PromotionGateError(Exception):
    """Base for promotion-gate errors raised by the controller."""


class PromotionGateDenied(PromotionGateError):
    """Raised when the promotion gate blocks a transition.

    Attributes:
        outcome: The :class:`PromotionGateReport` that caused the
            denial.  Always has ``passed=False`` with a populated
            ``failures`` tuple.
    """

    def __init__(self, message: str, outcome: PromotionGateReport) -> None:
        super().__init__(message)
        self.outcome = outcome


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _overrepresentation(v: dict[str, Any]) -> tuple[float | None, float | None]:
    ratio = v.get("phenom_overrepresentation")
    if ratio is None:
        ratio = v.get("overrepresentation_ratio")
    delta = v.get("overrepresentation_delta")
    if delta is None:
        delta = v.get("overrepresentation_delta_pp")

    if ratio is None:
        share_top = v.get("limited_share_top_decile")
        share_pool = v.get("limited_share_pool")
        if (
            isinstance(share_top, (int, float))
            and isinstance(share_pool, (int, float))
            and share_pool > 0
        ):
            ratio = float(share_top) / float(share_pool)
            delta = float(share_top) - float(share_pool)

    return (
        float(ratio) if isinstance(ratio, (int, float)) else None,
        float(delta) if isinstance(delta, (int, float)) else None,
    )


def _check_variant(
    report: dict[str, Any],
    *,
    variant: str,
    control: str,
    max_phenom_leakage: float,
    max_mae_delta_pct: float,
    max_overrep_delta_pp: float,
    require_cohort_keys: bool,
    require_canary_clean: bool,
) -> list[str]:
    failures: list[str] = []
    variants = report.get("variants") or {}
    v = variants.get(variant)
    if not v:
        return [f"variant {variant!r} missing from report"]
    if v.get("status") != "ok":
        return [f"variant {variant!r} status={v.get('status')!r} (expected ok)"]

    if require_cohort_keys:
        for key in REQUIRED_REPORT_KEYS:
            if key not in v:
                failures.append(f"missing required gate key {key!r} on {variant}")

    remaining = v.get("canary_anomalies_remaining")
    if require_canary_clean:
        if remaining is None:
            failures.append(
                "canary_anomalies_remaining missing — promotion DENY (fail-closed)"
            )
        elif not isinstance(remaining, (int, float)):
            failures.append(
                f"canary_anomalies_remaining has invalid type {type(remaining)!r}"
            )
        elif int(remaining) > 0:
            failures.append(
                f"canary_anomalies_remaining={int(remaining)} > 0 — promotion DENY"
            )

    phenom = v.get("phenom_leakage_rate")
    if phenom is not None:
        if not isinstance(phenom, (int, float)):
            failures.append(f"phenom_leakage_rate invalid type {type(phenom)!r}")
        elif float(phenom) > max_phenom_leakage:
            failures.append(
                f"phenom_leakage_rate={float(phenom):.4f} exceeds max={max_phenom_leakage:.4f}"
            )

    _ratio, delta = _overrepresentation(v)
    if delta is not None:
        if abs(delta) > 1.0:
            if abs(delta) > max_overrep_delta_pp:
                failures.append(
                    f"overrepresentation_delta={delta:.2f}pp exceeds "
                    f"+/-{max_overrep_delta_pp:.2f}pp"
                )
        else:
            if abs(delta) > (max_overrep_delta_pp / 100.0):
                failures.append(
                    f"overrepresentation_delta={delta:.4f} exceeds "
                    f"+/-{max_overrep_delta_pp / 100.0:.4f}"
                )

    control_v = variants.get(control) or {}
    control_mae = control_v.get("mae")
    variant_mae = v.get("mae")
    if (
        isinstance(control_mae, (int, float))
        and isinstance(variant_mae, (int, float))
        and control_mae > 0
    ):
        delta_pct = 100.0 * (float(variant_mae) - float(control_mae)) / float(control_mae)
        if delta_pct > max_mae_delta_pct:
            failures.append(
                f"MAE delta vs {control}: {delta_pct:+.2f}% exceeds "
                f"+{max_mae_delta_pct:.2f}% (control={control_mae}, variant={variant_mae})"
            )

    limited_mae = (v.get("mae_by_cohort") or {}).get("LIMITED")
    standard_mae = (v.get("mae_by_cohort") or {}).get("STANDARD")
    if (
        isinstance(limited_mae, (int, float))
        and isinstance(standard_mae, (int, float))
        and standard_mae > 0
        and limited_mae > 3.0 * standard_mae
    ):
        failures.append(
            f"LIMITED MAE ({limited_mae:.4f}) is >3x STANDARD MAE ({standard_mae:.4f}) "
            "— investigate before promoting"
        )

    return failures


# ── Programmatic entrypoint (WS6) ─────────────────────────────────────────


def evaluate_report(
    report_path: Path,
    *,
    variant: str = DEFAULT_VARIANT,
    control: str = CONTROL_VARIANT,
    max_phenom_leakage: float = 0.25,
    max_mae_delta_pct: float = 3.0,
    max_overrep_delta_pp: float = 5.0,
    require_cohort_keys: bool = True,
    require_canary_clean: bool = True,
    config_snapshot: Path | None = None,
) -> PromotionGateReport:
    """Evaluate a promotion report and return a :class:`PromotionGateReport`.

    Behavioural contract (WS6 — fail-closed):

    * I/O / JSON errors **propagate** as exceptions (the caller is
      responsible for surfacing them as a 2 / 500-style outcome).
    * Logic errors (missing keys, canary anomalies, leakage, MAE
      regression, config-hash mismatch) are collected into
      ``failures`` and the report is returned with ``passed=False``.
    * ``passed`` is True **iff** ``failures`` is empty and the report
      was readable.

    This function never raises :class:`PromotionGateDenied`; that is
    the controller's job.  It is the lowest-level structured
    evaluator and the right place to add new checks.
    """
    report = _load_report(report_path)
    failures = list(
        _check_variant(
            report,
            variant=variant,
            control=control,
            max_phenom_leakage=max_phenom_leakage,
            max_mae_delta_pct=max_mae_delta_pct,
            max_overrep_delta_pp=max_overrep_delta_pp,
            require_cohort_keys=require_cohort_keys,
            require_canary_clean=require_canary_clean,
        )
    )

    config_hash: str | None = None
    config_hash_status: str | None = None
    snapshot_path_str: str | None = None
    if config_snapshot is not None:
        snapshot_path_str = str(config_snapshot)
        snapshot = json.loads(config_snapshot.read_text(encoding="utf-8"))
        config_hash = compute_config_hash(snapshot)
        reported_hash = report.get("config_hash")
        if reported_hash is None:
            config_hash_status = "report_missing_config_hash"
        elif verify_config_hash(snapshot, reported_hash):
            config_hash_status = "match"
        else:
            config_hash_status = "mismatch"
            failures.append(
                f"config_hash mismatch: report={reported_hash!r} "
                f"candidate={config_hash!r} — promotion DENY (plan §18)"
            )

    return PromotionGateReport(
        passed=len(failures) == 0,
        failures=tuple(failures),
        report_path=str(report_path),
        variant=variant,
        control=control,
        config_hash=config_hash,
        config_hash_status=config_hash_status,
        config_snapshot_path=snapshot_path_str,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to harness report.json")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--control", default=CONTROL_VARIANT)
    parser.add_argument("--max-phenom-leakage", type=float, default=0.25)
    parser.add_argument("--max-mae-delta-pct", type=float, default=3.0)
    parser.add_argument("--max-overrep-delta-pp", type=float, default=5.0)
    parser.add_argument("--require-cohort-keys", action="store_true", default=True)
    parser.add_argument("--no-require-cohort-keys", action="store_false", dest="require_cohort_keys")
    parser.add_argument("--require-canary-clean", action="store_true", default=True)
    parser.add_argument("--no-require-canary-clean", action="store_false", dest="require_canary_clean")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument(
        "--config-snapshot",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file describing the candidate config.  When "
            "provided, its canonical SHA-256 is computed and compared "
            "against the report's recorded config_hash (plan §18).  "
            "Mismatch forces exit 1."
        ),
    )
    args = parser.parse_args(argv)

    try:
        outcome = evaluate_report(
            args.report,
            variant=args.variant,
            control=args.control,
            max_phenom_leakage=args.max_phenom_leakage,
            max_mae_delta_pct=args.max_mae_delta_pct,
            max_overrep_delta_pp=args.max_overrep_delta_pp,
            require_cohort_keys=args.require_cohort_keys,
            require_canary_clean=args.require_canary_clean,
            config_snapshot=args.config_snapshot,
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    failures = list(outcome.failures)
    summary: dict[str, Any] = {
        "variant": args.variant,
        "passed": outcome.passed,
        "failures": failures,
    }
    if outcome.config_hash is not None:
        summary["config_hash"] = {
            "candidate_config_hash": outcome.config_hash,
            "snapshot_path": outcome.config_snapshot_path,
            "report_config_hash": None
            if outcome.config_hash_status == "report_missing_config_hash"
            else (
                None
                if outcome.config_hash_status == "skipped"
                else _load_report(args.report).get("config_hash")
            ),
            "status": outcome.config_hash_status,
        }

    if not outcome.passed:
        print(f"PROMOTION GATE FAILED for variant={args.variant}:")
        for f in failures:
            print(f"  - {f}")
        if args.json_summary:
            print(json.dumps(summary, indent=2))
        return 1

    print(f"PROMOTION GATE PASSED for variant={args.variant}")
    if args.json_summary:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
