"""PR5 — single source of truth for ML training defaults.

Acceptance criteria (plan §20 / §33):
  API default == workflow default == MLConfig effective default

Canonical source: MLConfig (ml/config.py).
The GitHub workflow_dispatch inputs and the API dispatch payload
(_DEFAULT_WORKFLOW_INPUTS) must mirror the values that affect training
reproducibility: min_minutes, seed (random_seed), test_seasons, league.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _workflow_input_default(yaml_text: str, key: str) -> str | None:
    """Extract `default:` for a workflow_dispatch input by name (simple scan)."""
    pattern = rf"(?m)^      {re.escape(key)}:\n(?:.*\n)*?^        default:\s*\"?([^\n\"]+)\"?"
    m = re.search(pattern, yaml_text)
    return m.group(1).strip() if m else None


def _api_default(py_text: str, key: str) -> str | None:
    """Extract a value from _DEFAULT_WORKFLOW_INPUTS dict literal."""
    pattern = rf'"{re.escape(key)}":\s*"([^"]*)"'
    block = re.search(
        r"_DEFAULT_WORKFLOW_INPUTS[^=]*=\s*\{(.*?)\n\}",
        py_text,
        re.S,
    )
    if not block:
        return None
    m = re.search(pattern, block.group(1))
    return m.group(1) if m else None


@pytest.fixture(scope="module")
def sources() -> dict:
    # MLConfig() is instantiated at module import and requires database_url
    os.environ.setdefault("ML_DATABASE_URL", "postgresql://x:x@localhost/x")
    from ml.config import MLConfig

    cfg = MLConfig(database_url=os.environ["ML_DATABASE_URL"])

    workflow = _read(REPO_ROOT / ".github/workflows/ml-training.yml")
    api = _read(REPO_ROOT / "api/src/routers/ml_pipeline.py")
    return {
        "ml_min_minutes": cfg.min_minutes,
        "ml_seed": cfg.random_seed,
        "ml_test_seasons": cfg.test_seasons,
        "ml_league": cfg.league_name,
        "wf_min_minutes": _workflow_input_default(workflow, "min_minutes"),
        "wf_seed": _workflow_input_default(workflow, "seed"),
        "wf_test_seasons": _workflow_input_default(workflow, "test_seasons"),
        "wf_league": _workflow_input_default(workflow, "league"),
        "api_min_minutes": _api_default(api, "min_minutes"),
        "api_seed": _api_default(api, "seed"),
        "api_test_seasons": _api_default(api, "test_seasons"),
        "api_league": _api_default(api, "league"),
    }


def test_min_minutes_aligned(sources):
    """The bug that motivated PR5: API/workflow were 100, MLConfig was 800."""
    assert sources["ml_min_minutes"] == 800
    assert sources["wf_min_minutes"] == "800"
    assert sources["api_min_minutes"] == "800"


def test_seed_aligned(sources):
    assert sources["ml_seed"] == 42
    assert sources["wf_seed"] == "42"
    assert sources["api_seed"] == "42"


def test_test_seasons_aligned(sources):
    assert str(sources["ml_test_seasons"]) == sources["wf_test_seasons"] == sources["api_test_seasons"]


def test_league_aligned(sources):
    assert sources["ml_league"] == sources["wf_league"] == sources["api_league"] == "Serie A"


def test_no_stale_diverging_defaults_in_api():
    """Guard against regressing the old diverging defaults."""
    api = _read(REPO_ROOT / "api/src/routers/ml_pipeline.py")
    block = re.search(
        r"_DEFAULT_WORKFLOW_INPUTS[^=]*=\s*\{(.*?)\n\}",
        api,
        re.S,
    )
    assert block is not None
    assert '"min_minutes": "100"' not in block.group(1)
    assert '"seed": "4642"' not in block.group(1)
