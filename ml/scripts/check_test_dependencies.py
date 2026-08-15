#!/usr/bin/env python3
"""Preflight check for critical test/runtime dependencies.

Exits non-zero with a clear message if a required package is missing.
Used by CI and local developers before running the full suite.

Usage:
    python -m ml.scripts.check_test_dependencies
    # or
    python ml/scripts/check_test_dependencies.py
"""

from __future__ import annotations

import importlib
import sys

# (import_name, pip_package, min_version_hint)
REQUIRED = [
    ("pulp", "pulp", ">=2.7"),
    ("numpy", "numpy", None),
    ("pandas", "pandas", None),
    ("pydantic", "pydantic", None),
    ("sklearn", "scikit-learn", None),
]


def check() -> int:
    missing: list[tuple[str, str, str | None]] = []
    for import_name, pip_name, version_hint in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name, version_hint))

    if not missing:
        print("All critical test dependencies are present.")
        return 0

    print("ERROR: Missing critical dependencies required for the full test suite:\n")
    for import_name, pip_name, version_hint in missing:
        ver = f" ({version_hint})" if version_hint else ""
        print(f"  - package: {pip_name}{ver}")
        print(f"    import name: {import_name}")
        print(f"    install:     pip install '{pip_name}{version_hint or ''}'")
        print()
    print("Install the missing packages and re-run this check.")
    return 1


if __name__ == "__main__":
    sys.exit(check())
