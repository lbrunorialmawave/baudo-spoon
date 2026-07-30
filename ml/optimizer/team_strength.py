"""Load and normalize team strength (Elo) scores for inflation adjustment."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "team_strength_elo.json"


def _download_from_r2(path: Path) -> None:
    """Download team_strength_elo.json from R2 if configured."""
    from ml.config import settings

    if not settings.r2_endpoint_url:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
        )
        client.download_file(settings.r2_bucket_name, path.name, str(path))
        log.info("Downloaded from R2: %s", path.name)
    except Exception as exc:  # noqa: BLE001
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
