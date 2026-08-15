#!/usr/bin/env python3
"""Operational promotion gate for low-sample feature flags (WS4.5).

Reads an experiment harness ``report.json`` and exits non-zero when
cohort-aware quality gates fail. Intended to be run manually or in CI
*before* ``RolloutController.promote(new_stage=ACTIVE)`` for flags in
the LIMITED-sample family.

Usage::

    python -m ml.scripts.check_promotion_gate path/to/experiments/<run_id>/report.json
    python -m ml.scripts.check_promotion_gate report.json --variant C_shrinkage \\
        --max-phenom-leakage 0.25 --max-mae-delta-pct 3.0

Exit codes:
    0 — gates passed
    1 — gates failed
    2 — usage / I/O error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_VARIANT = "C_shrinkage"
CONTROL_VARIANT = "A_control"


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_variant(
    report: dict[str, Any],
    *,
    variant: str,
    control: str,
    max_phenom_leakage: float,
    max_mae_delta_pct: float,
    require_cohort_keys: bool,
) -> list[str]:
    failures: list[str] = []
    variants = report.get("variants") or {}
    v = variants.get(variant)
    if not v:
        return [f"variant {variant!r} missing from report"]
    if v.get("status") != "ok":
        return [f"variant {variant!r} status={v.get('status')!r} (expected ok)"]

    if require_cohort_keys:
        for key in ("mae_by_cohort", "rmse_by_cohort", "phenom_leakage_rate"):
            if key not in v:
                failures.append(f"missing cohort gate key {key!r} on {variant}")

    phenom = v.get("phenom_leakage_rate")
    if phenom is not None and phenom > max_phenom_leakage:
        failures.append(
            f"phenom_leakage_rate={phenom:.4f} exceeds max={max_phenom_leakage:.4f}"
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
            f"LIMITED MAE ({limited_mae:.4f}) is >3× STANDARD MAE ({standard_mae:.4f}) "
            "— investigate before promoting"
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to harness report.json")
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=f"Variant to gate (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--control",
        default=CONTROL_VARIANT,
        help=f"Control variant for MAE delta (default: {CONTROL_VARIANT})",
    )
    parser.add_argument(
        "--max-phenom-leakage",
        type=float,
        default=0.25,
        help="Max allowed LIMITED top-decile leakage rate (default: 0.25)",
    )
    parser.add_argument(
        "--max-mae-delta-pct",
        type=float,
        default=3.0,
        help="Max allowed aggregate MAE increase vs control in %% (default: 3.0)",
    )
    parser.add_argument(
        "--require-cohort-keys",
        action="store_true",
        default=True,
        help="Fail if mae_by_cohort / phenom_leakage_rate keys are absent",
    )
    parser.add_argument(
        "--no-require-cohort-keys",
        action="store_false",
        dest="require_cohort_keys",
    )
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
        require_cohort_keys=args.require_cohort_keys,
    )
    if failures:
        print(f"PROMOTION GATE FAILED for variant={args.variant}:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PROMOTION GATE PASSED for variant={args.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
