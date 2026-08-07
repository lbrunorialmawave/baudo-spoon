"""Load and normalize team strength (Elo) scores for inflation adjustment.

R2 download goes through ``ArtifactStore`` (the single boto3 entry-point);
see design doc "R2 come source of truth per gli artefatti ML/MANTRA"
(2026-08-02), Fase 5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "team_strength_elo.json"
)


def _download_from_r2(path: Path) -> None:
    """Ensure ``path`` is present locally, fetching from R2 via ArtifactStore if needed.

    Best-effort: any failure (missing credentials, import error, network) is
    logged and swallowed so the caller can fall back to an empty scores dict.
    """
    try:
        from ml.config import settings
        from ml.storage.artifact_store import ArtifactStore, R2Config
    except Exception as exc:
        log.warning("R2 download skipped for %s (import): %s", path.name, exc)
        return

    try:
        store = ArtifactStore(
            local_dir=path.parent,
            r2_config=R2Config(
                endpoint_url=settings.r2_endpoint_url,
                access_key_id=settings.r2_access_key_id,
                secret_access_key=settings.r2_secret_access_key,
                bucket_name=settings.r2_bucket_name,
            ),
        )
        # load_json downloads into local_dir if missing; we only care about the
        # side-effect of materialising the file (caller re-reads path).
        store.load_json(path.name)
    except Exception as exc:
        log.warning("R2 download skipped for %s: %s", path.name, exc)


def load_team_strength_scores(
    path: Path | None = None,
    known_teams: set[str] | None = None,
) -> dict[str, float]:
    """Load Elo data and min-max normalize to [0, 1].

    Parameters
    ----------
    path:
        Path to the JSON file. Defaults to ``config/team_strength_elo.json``.
    known_teams:
        If provided, only include clubs present in this set (filters out
        Serie B clubs not in the current league).

    Returns
    -------
    dict mapping club name → normalized score in [0, 1].
    """
    path = path or _DEFAULT_PATH
    if not path.exists():
        _download_from_r2(path)
    if not path.exists():
        log.warning("Team strength file not found at %s, returning empty scores", path)
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    clubs: dict[str, int] = raw.get("clubs", {})

    if known_teams:
        clubs = {k: v for k, v in clubs.items() if k in known_teams}

    if not clubs:
        return {}

    min_elo = min(clubs.values())
    max_elo = max(clubs.values())
    span = max_elo - min_elo
    if span == 0:
        return {k: 0.0 for k in clubs}

    return {k: (v - min_elo) / span for k, v in clubs.items()}
