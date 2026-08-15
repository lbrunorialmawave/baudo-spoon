"""WS12 observability metrics tests."""

from __future__ import annotations

import pytest

from ml.rollout.observability import (
    compute_cohort_observability,
    diagnostic_score_layers,
)


def test_compute_cohort_observability_counts():
    players = [
        {"sample_cohort": "LIMITED", "minutes": 200, "reliability_weight": 0.5, "projected_score": 8.0},
        {"sample_cohort": "LIMITED", "minutes": 400, "reliability_weight": 0.7, "projected_score": 7.0},
        {"sample_cohort": "STANDARD", "minutes": 900, "reliability_weight": 1.0, "projected_score": 6.0},
        {"sample_cohort": "STANDARD", "minutes": 1000, "reliability_weight": 1.0, "projected_score": 5.0},
        {"sample_cohort": "STANDARD", "minutes": 1100, "reliability_weight": 1.0, "projected_score": 4.0},
        {"sample_cohort": "STANDARD", "minutes": 1200, "reliability_weight": 1.0, "projected_score": 3.0},
        {"sample_cohort": "STANDARD", "minutes": 1300, "reliability_weight": 1.0, "projected_score": 2.0},
        {"sample_cohort": "STANDARD", "minutes": 1400, "reliability_weight": 1.0, "projected_score": 1.0},
        {"sample_cohort": "STANDARD", "minutes": 1500, "reliability_weight": 1.0, "projected_score": 0.5},
        {"sample_cohort": "INSUFFICIENT", "minutes": 50, "reliability_weight": 0.3, "projected_score": 9.0},
    ]
    snap = compute_cohort_observability(
        players,
        auction_reliability_enabled=True,
        optimizer_reliability_enabled=True,
        rollout_stage="shadow",
    )
    assert snap.limited_players_count == 2
    assert snap.standard_players_count == 7
    assert snap.insufficient_players_count == 1
    assert snap.mean_reliability_weight is not None
    assert snap.mean_minutes_limited == pytest.approx(300.0)
    assert snap.auction_reliability_enabled is True
    assert snap.rollout_stage == "shadow"
    d = snap.to_dict()
    assert "limited_players_count" in d


import pytest

def test_diagnostic_layers():
    layers = diagnostic_score_layers(raw_model=8.0, display=7.5, decision=5.0)
    assert layers["raw_model"] == 8.0
    assert layers["display"] == 7.5
    assert layers["decision"] == 5.0
