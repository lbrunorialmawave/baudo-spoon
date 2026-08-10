from __future__ import annotations

import pandas as pd

from ml.data.import_quotations import _fuzzy_match_one
from ml.mantra_ibrido.config import MantraIbridoConfig
from ml.mantra_ibrido.scoring import compute_hybrid_scores


def test_fuzzy_team_mismatch_is_explicitly_downgraded():
    candidates = pd.DataFrame([
        {
            "player_fotmob_id": 123,
            "player_name": "Mario Rossi",
            "team_fotmob": "Inter",
            "team_norm": "inter",
            "canonical_role": "FWD",
            "last_name_norm": "rossi",
        }
    ])
    result = _fuzzy_match_one("rossi", "roma", "FWD", candidates)
    assert result is not None
    _, _, _, score, team_mismatch = result
    assert team_mismatch is True
    assert score < 0.90


def test_foreign_fallback_confidence_is_penalized():
    base = {
        "predicted_fantavoto": 6.5,
        "prediction_std": 1.5,
        "expected_minutes": 1800,
        "FP_Corr": 55,
        "ruolo_primario": "FWD",
        "has_ml_data": True,
    }
    normal = compute_hybrid_scores([dict(base, is_foreign_fallback=False)], MantraIbridoConfig())[0]
    foreign = compute_hybrid_scores([dict(base, is_foreign_fallback=True)], MantraIbridoConfig())[0]
    assert foreign["confidenceScore"] < normal["confidenceScore"]
    assert "foreign_fallback" in foreign["hybridLabels"]


def test_foreign_fallback_cannot_enter_non_foreign_top_decile():
    players = []
    for i in range(10):
        players.append({
            "predicted_fantavoto": 6.0 + i * 0.1,
            "prediction_std": 0.2,
            "expected_minutes": 2000,
            "FP_Corr": 50,
            "ruolo_primario": "FWD",
            "has_ml_data": True,
            "is_foreign_fallback": False,
        })
    foreign = {
        "predicted_fantavoto": 9.0,
        "prediction_std": 0.0,
        "expected_minutes": 2700,
        "FP_Corr": 90,
        "ruolo_primario": "FWD",
        "has_ml_data": True,
        "is_foreign_fallback": True,
    }
    result = compute_hybrid_scores(players + [foreign], MantraIbridoConfig())
    foreign_conf = result[-1]["confidenceScore"]
    normal_conf = sorted(p["confidenceScore"] for p in result[:-1])
    assert foreign_conf < normal_conf[-1]
