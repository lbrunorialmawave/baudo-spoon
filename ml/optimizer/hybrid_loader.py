"""Load the MANTRA-ibrido ``fpIbrido`` signal for use in the CLASSIC objective.

Fase 4.6 (Unificazione CLASSIC ↔ segnali hybrid MANTRA): the ``fpIbrido``
score (weighted blend of MANTRA pillars + ML prediction, see
``ml/mantra_ibrido/scoring.py``) was only ever surfaced via the MANTRA-only
``/predictions/hybrid`` endpoints. This loader lets the CLASSIC optimizer
path optionally pull the same signal into ``Player.fp_ibrido`` and blend it
into the objective via ``OptimizationConfig.hybrid_blend`` — exactly the same
shape as the existing ``var_blend`` mechanism.

Search order (first hit wins), mirroring ``residual_loader.py``:
1. Local ``{artifacts_dir}/mantra_ibrido_results_{season}.json`` for
   season in (2026, 2025, 2024).
2. Same filenames via ``ArtifactStore`` (R2 fallback, downloads into cache).
3. Empty map → caller disables the blend with a warning.

``fpIbrido`` is stored 0-100; it is rescaled here to the 4-10 "voto" scale
that ``projected_score`` already uses (same formula ``compute_hybrid_scores``
itself applies for ``expectedValue``), so the blend in the solver is a
straightforward convex combination.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "HybridLoadReport",
    "load_fp_ibrido_map_from_artifacts",
    "load_fp_ibrido_map_preferring_r2",
]

_SEASONS = (2026, 2025, 2024)
_FANTAVOTO_MIN = 4.0
_FANTAVOTO_RANGE = 6.0  # matches ml/mantra_ibrido/scoring.py fp_ibrido_voto conversion


def _fp_ibrido_to_voto(fp_ibrido_0_100: float) -> float:
    return _FANTAVOTO_MIN + (fp_ibrido_0_100 / 100.0) * _FANTAVOTO_RANGE


def _rows_to_map(payload: Any) -> dict[str, float]:
    players = payload.get("players", []) if isinstance(payload, dict) else []
    out: dict[str, float] = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        fp = p.get("fpIbrido")
        fid = p.get("player_fotmob_id")
        if fp is None or fid is None:
            continue
        try:
            out[f"fm-{int(fid)}"] = _fp_ibrido_to_voto(float(fp))
        except (TypeError, ValueError):
            continue
    return out


class HybridLoadReport:
    def __init__(self, scores: dict[str, float], source: str, season: int | None, warnings: list[str] | None = None) -> None:
        self.scores = scores
        self.source = source
        self.season = season
        self.warnings = warnings or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "season": self.season,
            "n_players": len(self.scores),
            "warnings": list(self.warnings),
        }


def load_fp_ibrido_map_from_artifacts(artifacts_dir: str | Path | None) -> HybridLoadReport:
    """Try local ``mantra_ibrido_results_{season}.json`` files under *artifacts_dir*."""
    if not artifacts_dir:
        return HybridLoadReport({}, "none", None, ["artifacts_dir not set"])
    root = Path(artifacts_dir)
    for season in _SEASONS:
        path = root / f"mantra_ibrido_results_{season}.json"
        if not path.exists():
            continue
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            scores = _rows_to_map(payload)
            if scores:
                return HybridLoadReport(scores, f"local:{path.name}", season)
        except Exception as exc:  # noqa: BLE001
            log.warning("hybrid_loader_local_parse_failed path=%s err=%s", path, exc)
    return HybridLoadReport({}, "not_found", None, [f"no mantra_ibrido_results_*.json under {root}"])


def load_fp_ibrido_map_preferring_r2(
    artifacts_dir: str | Path | None = None,
    *,
    r2_endpoint_url: str | None = None,
    r2_access_key_id: str | None = None,
    r2_secret_access_key: str | None = None,
    r2_bucket_name: str | None = None,
) -> HybridLoadReport:
    """Local disk first, then R2 via ArtifactStore, else empty map with warnings."""
    art = artifacts_dir or os.environ.get("API_ARTIFACTS_DIR") or os.environ.get("ARTIFACTS_DIR")
    local = load_fp_ibrido_map_from_artifacts(art)
    if local.scores:
        return local

    try:
        from ml.storage.artifact_store import ArtifactStore, R2Config
    except Exception as exc:  # noqa: BLE001
        return HybridLoadReport({}, "r2_unavailable", None, local.warnings + [f"ArtifactStore import failed: {exc}"])

    root = Path(art) if art else Path("artifacts")
    root.mkdir(parents=True, exist_ok=True)

    endpoint = r2_endpoint_url or os.environ.get("API_R2_ENDPOINT_URL") or os.environ.get("ML_R2_ENDPOINT_URL")
    key_id = r2_access_key_id or os.environ.get("API_R2_ACCESS_KEY_ID") or os.environ.get("ML_R2_ACCESS_KEY_ID")
    secret = r2_secret_access_key or os.environ.get("API_R2_SECRET_ACCESS_KEY") or os.environ.get("ML_R2_SECRET_ACCESS_KEY")
    bucket = r2_bucket_name or os.environ.get("API_R2_BUCKET_NAME") or os.environ.get("ML_R2_BUCKET_NAME")

    r2_cfg = None
    if endpoint and key_id and secret and bucket:
        r2_cfg = R2Config(endpoint_url=endpoint, access_key_id=key_id, secret_access_key=secret, bucket_name=bucket)

    store = ArtifactStore(local_dir=root, r2_config=r2_cfg)
    for season in _SEASONS:
        filename = f"mantra_ibrido_results_{season}.json"
        data = store.load_json(filename)
        if data is None:
            continue
        scores = _rows_to_map(data)
        if scores:
            return HybridLoadReport(scores, f"r2_or_local:{filename}", season, local.warnings)

    return HybridLoadReport({}, "not_found_local_or_r2", None, local.warnings + ["no mantra_ibrido_results_*.json found locally or on R2"])
