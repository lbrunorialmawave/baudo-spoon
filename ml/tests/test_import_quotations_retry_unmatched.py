"""Unit tests for retry_unmatched (P2 neo-arrivi coverage)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml.data.import_quotations import retry_unmatched


def _mapping_row(
    fantacalcio_id: int = 1,
    name: str = "Mario Rossi",
    team: str = "Roma",
    role: str | None = "FWD",
):
    return {
        "fantacalcio_id": fantacalcio_id,
        "season_start": 2025,
        "name_fantacalcio": name,
        "team_fantacalcio": team,
        "canonical_role": role,
        "match_method": "unmatched",
    }


def test_retry_unmatched_accepts_single_candidate():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    conn.execute.return_value.mappings.return_value.all.return_value = [
        _mapping_row()
    ]

    single = [{"id": 4242, "name": "Mario Rossi", "team_name": "AS Roma", "score": 1.0}]

    with patch("ml.data.import_quotations._fotmob_suggest_api", return_value=single):
        with patch("ml.data.import_quotations.persist_player_id_map", return_value=1) as persist:
            df = retry_unmatched(engine, 2025)

    assert len(df) == 1
    assert int(df.iloc[0]["player_fotmob_id"]) == 4242
    assert df.iloc[0]["match_method"] == "fotmob_suggest_retry"
    assert float(df.iloc[0]["confidence"]) == pytest.approx(0.85)
    persist.assert_called_once()


def test_retry_unmatched_rejects_zero_candidates():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    conn.execute.return_value.mappings.return_value.all.return_value = [
        _mapping_row(name="Unknown Nobody")
    ]

    with patch("ml.data.import_quotations._fotmob_suggest_api", return_value=[]):
        with patch("ml.data.import_quotations.persist_player_id_map") as persist:
            df = retry_unmatched(engine, 2025)

    assert df.empty
    persist.assert_not_called()


def test_retry_unmatched_rejects_multiple_candidates():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    conn.execute.return_value.mappings.return_value.all.return_value = [
        _mapping_row(name="Rossi")
    ]

    multi = [
        {"id": 1, "name": "Rossi A", "team_name": "X", "score": 1.0},
        {"id": 2, "name": "Rossi B", "team_name": "Y", "score": 0.9},
    ]

    with patch("ml.data.import_quotations._fotmob_suggest_api", return_value=multi):
        with patch("ml.data.import_quotations.persist_player_id_map") as persist:
            df = retry_unmatched(engine, 2025)

    assert df.empty
    persist.assert_not_called()


def test_retry_unmatched_no_rows():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.all.return_value = []

    df = retry_unmatched(engine, 2025)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
