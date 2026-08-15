"""Synthetic canary dataset for LIMITED-cohort hardening regression tests.

WS0 of plan-limited-cohort-hardening.md.

Contains a small, deterministic set of LIMITED players (including an
Adzic-style outlier: ~163 minutes, 1 goal → extreme per-90) plus a
handful of STANDARD reference players per role.  No live DB dependency —
safe for CI.

Columns mirror the shape consumed by output-reliability / decision weight
paths so the same fixture can guard WS1 (shrinkage), WS2 (continuous
weight) and WS3 (Optimizer/Auction alignment).
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# Known anomaly marker used by harness / tests.
CANARY_ANOMALY_IDS: Final[frozenset[str]] = frozenset({"adzic-163"})


def build_limited_cohort_canary() -> pd.DataFrame:
    """Return a deterministic canary DataFrame.

    Rows with ``is_known_anomaly=True`` are the false-phenom cases that
    must leave the top bracket after shrinkage + continuous reliability
    weight are active.
    """
    rows = [
        # ── Adzic-style false phenom (LIMITED, extreme per-90) ──────────
        {
            "player_id": "adzic-163",
            "player_name": "Adzic (canary)",
            "canonical_role": "FWD",
            "mins_played": 163,
            "goals": 1,
            "predicted_fantavoto": 8.2,  # artificially high raw (pre-shrink)
            "sample_cohort": "LIMITED",
            "is_known_anomaly": True,
            "cost": 12,
        },
        # ── Other LIMITED borderline cases ──────────────────────────────
        {
            "player_id": "lim-105",
            "player_name": "LowMinutes Forward",
            "canonical_role": "FWD",
            "mins_played": 105,
            "goals": 0,
            "predicted_fantavoto": 7.4,
            "sample_cohort": "LIMITED",
            "is_known_anomaly": True,
            "cost": 8,
        },
        {
            "player_id": "lim-795",
            "player_name": "NearStandard Mid",
            "canonical_role": "MID",
            "mins_played": 795,
            "goals": 3,
            "predicted_fantavoto": 6.8,
            "sample_cohort": "LIMITED",
            "is_known_anomaly": False,
            "cost": 15,
        },
        {
            "player_id": "lim-400",
            "player_name": "MidLimited Def",
            "canonical_role": "DEF",
            "mins_played": 400,
            "goals": 0,
            "predicted_fantavoto": 6.3,
            "sample_cohort": "LIMITED",
            "is_known_anomaly": False,
            "cost": 6,
        },
        # ── STANDARD reference players (high confidence) ────────────────
        {
            "player_id": "std-fwd-1",
            "player_name": "Top Forward A",
            "canonical_role": "FWD",
            "mins_played": 2800,
            "goals": 15,
            "predicted_fantavoto": 7.1,
            "sample_cohort": "STANDARD",
            "is_known_anomaly": False,
            "cost": 35,
        },
        {
            "player_id": "std-fwd-2",
            "player_name": "Top Forward B",
            "canonical_role": "FWD",
            "mins_played": 2500,
            "goals": 12,
            "predicted_fantavoto": 6.9,
            "sample_cohort": "STANDARD",
            "is_known_anomaly": False,
            "cost": 28,
        },
        {
            "player_id": "std-mid-1",
            "player_name": "Top Midfielder",
            "canonical_role": "MID",
            "mins_played": 3000,
            "goals": 6,
            "predicted_fantavoto": 6.7,
            "sample_cohort": "STANDARD",
            "is_known_anomaly": False,
            "cost": 22,
        },
        {
            "player_id": "std-def-1",
            "player_name": "Top Defender",
            "canonical_role": "DEF",
            "mins_played": 3100,
            "goals": 2,
            "predicted_fantavoto": 6.4,
            "sample_cohort": "STANDARD",
            "is_known_anomaly": False,
            "cost": 18,
        },
        {
            "player_id": "std-gk-1",
            "player_name": "Top Keeper",
            "canonical_role": "GK",
            "mins_played": 3200,
            "goals": 0,
            "predicted_fantavoto": 6.2,
            "sample_cohort": "STANDARD",
            "is_known_anomaly": False,
            "cost": 15,
        },
        # ── INSUFFICIENT (should stay heavily discounted) ───────────────
        {
            "player_id": "ins-40",
            "player_name": "AlmostNoMinutes",
            "canonical_role": "FWD",
            "mins_played": 40,
            "goals": 1,
            "predicted_fantavoto": 9.0,
            "sample_cohort": "INSUFFICIENT",
            "is_known_anomaly": True,
            "cost": 5,
        },
    ]
    return pd.DataFrame(rows)


def canary_anomaly_count(df: pd.DataFrame) -> int:
    """Return number of known-anomaly rows still present."""
    if "is_known_anomaly" not in df.columns:
        return 0
    return int(df["is_known_anomaly"].sum())
