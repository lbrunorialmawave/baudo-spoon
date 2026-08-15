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
from pathlib import Path
from typing import Any


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
    args = parser.parse_args(argv)

    try:
        report = _load_report(args.report)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    failures = _check_variant(
        report,
        variant=args.variant,
        control=args.control,
        max_phenom_leakage=args.max_phenom_leakage,
        max_mae_delta_pct=args.max_mae_delta_pct,
        max_overrep_delta_pp=args.max_overrep_delta_pp,
        require_cohort_keys=args.require_cohort_keys,
        require_canary_clean=args.require_canary_clean,
    )

    summary = {"variant": args.variant, "passed": len(failures) == 0, "failures": failures}

    if failures:
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
