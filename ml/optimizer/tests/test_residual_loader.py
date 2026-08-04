import json
from pathlib import Path

from ml.optimizer.job_store import MemoryJobStore
from ml.optimizer.models import Player
from ml.optimizer.residual_loader import (
    load_residuals_from_artifacts,
    load_residuals_from_path,
    merge_with_prediction_std,
)


def test_load_json(tmp_path: Path):
    p = tmp_path / "residuals.json"
    p.write_text(
        json.dumps(
            [
                {"player_id": "a", "role": "C", "residual": 0.5},
                {"playerId": "b", "role": "A", "actual": 7.0, "predicted": 6.5},
            ]
        )
    )
    rep = load_residuals_from_path(p)
    assert len(rep.residuals) == 2
    assert rep.n_players == 2


def test_artifacts_missing(tmp_path: Path):
    rep = load_residuals_from_artifacts(tmp_path)
    assert rep.residuals == []
    assert rep.source == "not_found"


def test_merge_prediction_std():
    pool = [
        Player(
            player_id="x",
            name="x",
            role="C",
            real_team="R",
            cost=10,
            projected_score=6.0,
            prediction_std=0.4,
        )
    ]
    merged = merge_with_prediction_std([], pool, random_seed=1, n_synthetic=5)
    assert len(merged) == 5


def test_memory_job_store():
    s = MemoryJobStore()
    j = s.create(request_meta={"n": 1})
    s.set_running(j.job_id)
    s.set_completed(j.job_id, result={"ok": True})
    assert s.get(j.job_id).status == "completed"
