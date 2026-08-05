"""Tests for the ML-prediction plumbing on the MANTRA artefact.

These tests guard the *informational* (non-reconciling) propagation of
``season_value`` / ``start_probability`` from the ML predictions
artefact onto each player record in ``mantra_results_{season}.json``.

Scope-boundary reminders (P1-4 is still open):

* The two fields sit alongside ``FP_Mantra`` / ``VR`` without
  blending, cross-validation, or any kind of reconciliation.
* Missing predictions yield ``None`` on both fields, matching the
  existing ``Fase7`` / ``rischio`` ``None`` pattern.
* The predictions artefact is read once per ``run_mantra`` call; its
  location defaults to the same ``artifacts_dir`` the trainer writes
  to, but a missing file is the common, non-fatal case (e.g. a fresh
  season where the trainer has not been run yet).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.mantra.runner import _load_predictions_by_id, run_mantra


# ── _load_predictions_by_id: file-level plumbing ────────────────────────────


def _write_predictions_artifact(directory: Path, payload: dict) -> Path:
    path = directory / "results_latest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_predictions_by_id_returns_empty_when_dir_is_none():
    """No ``output_dir`` → empty lookup (no info, no error)."""
    assert _load_predictions_by_id(None) == {}


def test_load_predictions_by_id_returns_empty_when_file_missing(tmp_path: Path):
    """Missing artefact file → empty lookup."""
    assert _load_predictions_by_id(tmp_path) == {}


def test_load_predictions_by_id_returns_empty_on_malformed_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Corrupted artefact → warning logged, empty lookup returned."""
    (tmp_path / "results_latest.json").write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="ml.mantra.runner"):
        result = _load_predictions_by_id(tmp_path)
    assert result == {}
    assert any("Could not read predictions artefact" in rec.message for rec in caplog.records)


def test_load_predictions_by_id_indexes_by_player_fotmob_id(tmp_path: Path):
    """A valid artefact is keyed by integer ``player_fotmob_id``."""
    _write_predictions_artifact(
        tmp_path,
        {
            "predictions": [
                {
                    "player_fotmob_id": 100,
                    "predicted_fantavoto": 7.0,
                    "expected_minutes": 2700.0,
                    "fantapunti_totali": 210.0,
                    "probabilita_titolarita": 0.79,
                },
                {
                    "player_fotmob_id": 200,
                    "predicted_fantavoto": 6.0,
                    "expected_minutes": 1500.0,
                    "fantapunti_totali": 100.0,
                    "probabilita_titolarita": 0.44,
                },
            ]
        },
    )
    lookup = _load_predictions_by_id(tmp_path)
    assert set(lookup.keys()) == {100, 200}
    assert lookup[100]["fantapunti_totali"] == pytest.approx(210.0)
    assert lookup[200]["probabilita_titolarita"] == pytest.approx(0.44)


def test_load_predictions_by_id_skips_records_without_fotmob_id(tmp_path: Path):
    """Records with no / NaN / non-numeric ``player_fotmob_id`` are dropped."""
    _write_predictions_artifact(
        tmp_path,
        {
            "predictions": [
                {"player_fotmob_id": None, "fantapunti_totali": 1.0},
                {"player_fotmob_id": float("nan"), "fantapunti_totali": 2.0},
                {"player_fotmob_id": "abc", "fantapunti_totali": 3.0},
                {"player_fotmob_id": 10, "fantapunti_totali": 4.0},
            ]
        },
    )
    lookup = _load_predictions_by_id(tmp_path)
    assert set(lookup.keys()) == {10}


# ── run_mantra: end-to-end propagation onto players_out ─────────────────────


def _stub_df() -> pd.DataFrame:
    """Two players: one with a FotMob id mapping, one without.

    Includes every column ``run_mantra`` reads from ``df`` (after the
    pre-computed columns ``load_data`` would add — we mock ``load_data``
    so the stub has to be self-contained).
    """
    df = pd.DataFrame(
        {
            "fantacalcio_id": [1, 2],
            "player_fotmob_id": [100, np.nan],
            "season_start": [2024, 2024],
            "player_name": ["Alpha", "Beta"],
            "team": ["AAA", "BBB"],
            "ruolo_primario": ["FWD", "MID"],
            "ruoli_mantra": [["FWD"], ["MID"]],
            "Pz1": [10, 5],
            "Pz2": [8, 4],
            "Pz3": [6, 3],
            "num_ruoli": [1, 1],
            "is_neo_arrivo": [False, False],
            "is_starter": [True, False],
            "stats_from_prior_season": [False, False],
            "stats_from_foreign_league": [False, False],
        }
    )
    return df


def _stub_pillar_series(value: float = 6.0) -> pd.Series:
    return pd.Series([value, value], index=[0, 1], dtype=float)


def _stub_scores(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "fp_corr": _stub_pillar_series(7.0),
        "cp_corr": _stub_pillar_series(8.0),
        "fp_mantra": _stub_pillar_series(7.5),
        "vr": _stub_pillar_series(9.0),
        "prezzo_massimo": _stub_pillar_series(15.0),
    }


def _patch_heavy_compute():
    """Patch the MANTRA pipeline's heavy compute steps with stubs."""
    return [
        patch("ml.mantra.runner.compute_p1", return_value=_stub_pillar_series(6.0)),
        patch("ml.mantra.runner.compute_p2", return_value=_stub_pillar_series(6.5)),
        patch("ml.mantra.runner.compute_p3", return_value=_stub_pillar_series(5.5)),
        patch("ml.mantra.runner.compute_p4", return_value=_stub_pillar_series(4.5)),
        patch("ml.mantra.runner.compute_cp", return_value=_stub_pillar_series(6.0)),
        patch("ml.mantra.runner.compute_fp", return_value=_stub_pillar_series(7.0)),
        patch("ml.mantra.runner.compute_fp_corr", return_value=_stub_scores(_stub_df())),
        patch(
            "ml.mantra.runner.classify_fase7",
            return_value=(
                pd.Series(["starter", "bench"]),
                pd.Series([None, None]),
            ),
        ),
        patch("ml.mantra.runner.top_per_ruolo", return_value={}),
        patch("ml.mantra.runner.multi_eleggibilita", return_value={}),
        patch(
            "ml.mantra.runner.low_cost",
            side_effect=lambda *_a, **_k: pd.DataFrame({"player_name": []}),
        ),
        patch("ml.mantra.runner.scommesse_multi_ruolo", return_value=pd.DataFrame({"player_name": []})),
        patch("ml.mantra.runner.watchlist_giovani", return_value=pd.DataFrame({"player_name": []})),
        patch("ml.mantra.runner.rischio_contestuale", return_value=pd.Series(["low", "high"])),
    ]


class _Patches:
    """Tiny RAII helper around a list of ``unittest.mock.patch`` contexts.

    Lets each test do ``with _Patches(...):`` without writing nested
    try/finally blocks for every test.
    """

    def __init__(self, *patches) -> None:
        self._patches = list(patches)
        self._stack = None

    def __enter__(self) -> "_Patches":
        from contextlib import ExitStack

        self._stack = ExitStack()
        for p in self._patches:
            self._stack.enter_context(p)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            self._stack.__exit__(exc_type, exc, tb)


def test_run_mantra_adds_season_value_and_start_probability_to_each_player(tmp_path: Path):
    """Two players, one matched in the artefact, one not.

    * Alpha (fotmob_id=100) is in the artefact → both fields populated.
    * Beta (fotmob_id=NaN) has no id-mapping → both fields ``None``,
      matching the existing ``Fase7`` / ``rischio`` ``None`` pattern.
    """
    df = _stub_df()
    _write_predictions_artifact(
        tmp_path,
        {
            "predictions": [
                {
                    "player_fotmob_id": 100,
                    "predicted_fantavoto": 7.0,
                    "expected_minutes": 2700.0,
                    "fantapunti_totali": 210.0,
                    "probabilita_titolarita": 0.79,
                }
            ]
        },
    )

    with _Patches(
        *_patch_heavy_compute(),
        patch("ml.mantra.runner.load_data", return_value=df),
        patch("ml.mantra.runner.compute_ps_corretto", return_value=pd.Series([50.0])),
    ):
        result = run_mantra(engine=None, season_start=2024, output_dir=tmp_path)

    players = result["players"]
    assert len(players) == 2

    alpha, beta = players
    # Alpha: matched prediction → both fields populated.
    assert alpha["player_fotmob_id"] == 100
    assert alpha["season_value"] == pytest.approx(210.0)
    assert alpha["start_probability"] == pytest.approx(0.79)

    # Beta: no id-mapping → both fields None, no crash, no leak.
    assert beta["player_fotmob_id"] is None
    assert beta["season_value"] is None
    assert beta["start_probability"] is None

    # The two fields are present on every record, never silently absent.
    for p in players:
        assert "season_value" in p
        assert "start_probability" in p


def test_run_mantra_falls_back_to_derivation_when_artifact_has_no_precomputed(
    tmp_path: Path,
):
    """Older artefacts (no pre-computed columns) still get a value pair,
    thanks to the derivation in ``resolve_season_value_fields``."""
    df = _stub_df()
    _write_predictions_artifact(
        tmp_path,
        {
            "predictions": [
                {
                    "player_fotmob_id": 100,
                    "predicted_fantavoto": 7.0,
                    "expected_minutes": 2700.0,
                }
            ]
        },
    )

    with _Patches(
        *_patch_heavy_compute(),
        patch("ml.mantra.runner.load_data", return_value=df),
        patch("ml.mantra.runner.compute_ps_corretto", return_value=pd.Series([50.0])),
    ):
        result = run_mantra(engine=None, season_start=2024, output_dir=tmp_path)

    alpha = result["players"][0]
    assert alpha["season_value"] == pytest.approx(210.0)  # 7.0 * 30
    assert alpha["start_probability"] == pytest.approx(2700.0 / 3420.0)


def test_run_mantra_returns_none_for_both_fields_when_artefact_missing(tmp_path: Path):
    """No ``results_latest.json`` → both fields are ``None`` for all players.

    This is the *common* case for a fresh season: the trainer has not
    been run yet. The MANTRA pipeline must not block on it.
    """
    df = _stub_df()
    assert not (tmp_path / "results_latest.json").exists()

    with _Patches(
        *_patch_heavy_compute(),
        patch("ml.mantra.runner.load_data", return_value=df),
        patch("ml.mantra.runner.compute_ps_corretto", return_value=pd.Series([50.0])),
    ):
        result = run_mantra(engine=None, season_start=2024, output_dir=tmp_path)

    for p in result["players"]:
        assert p["season_value"] is None
        assert p["start_probability"] is None


def test_run_mantra_persists_both_fields_to_disk(tmp_path: Path):
    """Acceptance gate: the JSON artefact written to disk carries both
    fields, so the ``/mantra/players`` endpoint can serve them without
    any further change.
    """
    df = _stub_df()
    _write_predictions_artifact(
        tmp_path,
        {
            "predictions": [
                {
                    "player_fotmob_id": 100,
                    "predicted_fantavoto": 7.0,
                    "expected_minutes": 2700.0,
                    "fantapunti_totali": 210.0,
                    "probabilita_titolarita": 0.79,
                }
            ]
        },
    )

    with _Patches(
        *_patch_heavy_compute(),
        patch("ml.mantra.runner.load_data", return_value=df),
        patch("ml.mantra.runner.compute_ps_corretto", return_value=pd.Series([50.0])),
    ):
        run_mantra(engine=None, season_start=2024, output_dir=tmp_path)

    on_disk = json.loads((tmp_path / "mantra_results_2024.json").read_text(encoding="utf-8"))
    for p in on_disk["players"]:
        assert "season_value" in p
        assert "start_probability" in p


def test_run_mantra_persists_without_ml_database_url_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regressione: la scrittura dell'artefatto (via ArtifactStore) non deve
    richiedere ML_DATABASE_URL — run_mantra riceve un ``engine`` già
    connesso (o None nei test) e non ha altrimenti bisogno di configurazione
    DB. Costruire l'R2Config passando per il singleton MLConfig/Settings
    introdurrebbe quella dipendenza indiretta; R2Config.from_env() no.
    """
    monkeypatch.delenv("ML_DATABASE_URL", raising=False)
    df = _stub_df()
    _write_predictions_artifact(tmp_path, {"predictions": []})

    with _Patches(
        *_patch_heavy_compute(),
        patch("ml.mantra.runner.load_data", return_value=df),
        patch("ml.mantra.runner.compute_ps_corretto", return_value=pd.Series([50.0])),
    ):
        run_mantra(engine=None, season_start=2024, output_dir=tmp_path)

    assert (tmp_path / "mantra_results_2024.json").exists()
