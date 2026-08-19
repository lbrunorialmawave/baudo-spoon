"""Contract test: frontend pitch catalog ↔ backend Mantra formations.

Prevents drift between modules drawn on the pitch and modules the optimizer
can solve (plan §13 — same idea as test_frontend_auction_contract).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ml.optimizer.formations import MANTRA_FORMATIONS, MANTRA_FORMATIONS_BY_LABEL

REPO_ROOT = Path(__file__).resolve().parents[3]
PITCH_TS = (
    REPO_ROOT
    / "frontend"
    / "src"
    / "app"
    / "core"
    / "constants"
    / "pitch-coordinates.ts"
)


def _parse_frontend_formation_labels(source: str) -> list[str]:
    """Extract string literals from MANTRA_FORMATION_LABELS array."""
    m = re.search(
        r"export const MANTRA_FORMATION_LABELS\s*=\s*\[(.*?)\]\s*as const",
        source,
        re.DOTALL,
    )
    if not m:
        pytest.fail("MANTRA_FORMATION_LABELS not found in pitch-coordinates.ts")
    return re.findall(r"'([^']+)'", m.group(1))


def test_pitch_coordinates_file_exists():
    assert PITCH_TS.is_file(), f"Missing frontend constant file: {PITCH_TS}"


def test_frontend_labels_match_backend_catalog():
    source = PITCH_TS.read_text(encoding="utf-8")
    fe = _parse_frontend_formation_labels(source)
    be = [f.label for f in MANTRA_FORMATIONS]

    assert fe == be, (
        "Formation label drift between frontend and backend.\n"
        f"  only frontend: {sorted(set(fe) - set(be))}\n"
        f"  only backend:  {sorted(set(be) - set(fe))}\n"
        f"  frontend order: {fe}\n"
        f"  backend order:  {be}"
    )


def test_backend_catalog_lookup():
    for f in MANTRA_FORMATIONS:
        assert MANTRA_FORMATIONS_BY_LABEL[f.label] is f


def test_tuned_layouts_are_subset_of_catalog():
    """Hand-tuned TUNED keys must be valid catalog labels."""
    source = PITCH_TS.read_text(encoding="utf-8")
    # keys inside const TUNED: Record<...> = { '4-3-3': ...
    m = re.search(r"const TUNED[^=]*=\s*\{(.*?)\n\};", source, re.DOTALL)
    if not m:
        pytest.skip("TUNED block not found — layout may be fully generic")
    keys = re.findall(r"^\s*'([^']+)':", m.group(1), re.MULTILINE)
    unknown = [k for k in keys if k not in MANTRA_FORMATIONS_BY_LABEL]
    assert not unknown, f"TUNED layouts for unknown formations: {unknown}"
