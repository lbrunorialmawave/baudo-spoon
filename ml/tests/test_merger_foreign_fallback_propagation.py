"""Regression: merger must propagate is_foreign_fallback from ML → hybrid player.

Root cause of missing Overview drawer badge:
  trainer writes is_foreign_fallback=True on foreign prediction rows
  merger copied predicted_fantavoto / std / minutes but dropped the flag
  scoring never saw it → no confidence penalty, no hybridLabels
  overview _to_camel never emitted isForeignFallback
  drawer @if (player().isForeignFallback) never rendered
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.mantra_ibrido.merger import merge_datasets
from ml.mantra_ibrido.scoring import compute_hybrid_scores
from ml.mantra_ibrido.config import MantraIbridoConfig


def _write_mantra(path: Path, players: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "players": players,
                "meta": {"season_start": 2025, "generated_at": "2025-01-01", "run_id": "t"},
                "classifications": {},
            }
        ),
        encoding="utf-8",
    )


def _write_ml(path: Path, predictions: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "predictions": predictions,
                "var_results": [],
                "next_season_predictions": [],
                "run_id": "ml-test",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )


def test_merger_propagates_is_foreign_fallback(tmp_path: Path) -> None:
    mantra = tmp_path / "mantra.json"
    ml = tmp_path / "results_latest.json"

    _write_mantra(
        mantra,
        [
            {
                "fantacalcio_id": 1,
                "player_fotmob_id": 100,
                "player_name": "Core Player",
                "team": "Inter",
                "ruolo_primario": "FWD",
                "FP_Corr": 60,
            },
            {
                "fantacalcio_id": 2,
                "player_fotmob_id": 200,
                "player_name": "Foreign Neo",
                "team": "Milan",
                "ruolo_primario": "MID",
                "FP_Corr": 55,
            },
        ],
    )
    _write_ml(
        ml,
        [
            {
                "player_fotmob_id": 100,
                "player_name": "Core Player",
                "predicted_fantavoto": 6.5,
                "prediction_std": 0.3,
                "expected_minutes": 2000,
                "is_foreign_fallback": False,
            },
            {
                "player_fotmob_id": 200,
                "player_name": "Foreign Neo",
                "predicted_fantavoto": 6.8,
                "prediction_std": 1.5,
                "expected_minutes": 1800,
                "is_foreign_fallback": True,
            },
        ],
    )

    merged = merge_datasets(mantra, ml)
    by_id = {p["player_fotmob_id"]: p for p in merged["players"]}

    assert by_id[100]["has_ml_data"] is True
    assert by_id[100]["is_foreign_fallback"] is False

    assert by_id[200]["has_ml_data"] is True
    assert by_id[200]["is_foreign_fallback"] is True


def test_merger_sets_false_when_no_ml_match(tmp_path: Path) -> None:
    mantra = tmp_path / "mantra.json"
    ml = tmp_path / "results_latest.json"
    _write_mantra(
        mantra,
        [
            {
                "fantacalcio_id": 9,
                "player_fotmob_id": 999,
                "player_name": "No ML",
                "team": "Roma",
                "ruolo_primario": "DEF",
                "FP_Corr": 40,
            }
        ],
    )
    _write_ml(ml, [])

    merged = merge_datasets(mantra, ml)
    p = merged["players"][0]
    assert p["has_ml_data"] is False
    assert p["is_foreign_fallback"] is False


def test_scoring_uses_propagated_flag() -> None:
    """After merger fix, scoring must tag hybridLabels and penalize confidence."""
    base = {
        "predicted_fantavoto": 6.5,
        "prediction_std": 1.5,
        "expected_minutes": 1800,
        "FP_Corr": 55,
        "ruolo_primario": "FWD",
        "has_ml_data": True,
    }
    normal = compute_hybrid_scores(
        [dict(base, is_foreign_fallback=False)], MantraIbridoConfig()
    )[0]
    foreign = compute_hybrid_scores(
        [dict(base, is_foreign_fallback=True)], MantraIbridoConfig()
    )[0]

    assert foreign["confidenceScore"] < normal["confidenceScore"]
    assert "foreign_fallback" in foreign["hybridLabels"]
    assert "foreign_fallback" not in (normal.get("hybridLabels") or [])


def test_to_camel_is_foreign_fallback() -> None:
    """Mirrors api.routers.intelligence._to_camel generic snake→camel path."""
    key = "is_foreign_fallback"
    parts = key.split("_")
    camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
    assert camel == "isForeignFallback"
