"""Load historical prediction residuals for Monte Carlo bootstrap.

Search order (first hit wins):
1. Explicit path / list passed by caller
2. ``{artifacts_dir}/residuals.json`` or ``residuals.parquet``
3. ``{artifacts_dir}/evaluation/residuals.*``
4. Empty list → caller falls back to prediction_std / parametric

JSON schema (list of objects):
  { "player_id": str, "role": str, "residual": float }
  residual = actual_fantavoto - predicted_fantavoto
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

__all__ = [
    "load_residuals_from_path",
    "load_residuals_from_artifacts",
    "load_residuals_preferring_r2",
    "ResidualLoadReport",
]


class ResidualLoadReport:
    def __init__(
        self,
        residuals: list[dict[str, Any]],
        source: str,
        n_players: int,
        n_roles: int,
        warnings: list[str] | None = None,
    ) -> None:
        self.residuals = residuals
        self.source = source
        self.n_players = n_players
        self.n_roles = n_roles
        self.warnings = warnings or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_rows": len(self.residuals),
            "n_players": self.n_players,
            "n_roles": self.n_roles,
            "warnings": list(self.warnings),
        }


def _normalize_row(row: dict) -> dict[str, Any] | None:
    pid = row.get("player_id") or row.get("playerId") or row.get("id")
    role = row.get("role") or row.get("ruolo") or "C"
    res = row.get("residual")
    if res is None and "actual" in row and "predicted" in row:
        try:
            res = float(row["actual"]) - float(row["predicted"])
        except (TypeError, ValueError):
            return None
    if pid is None or res is None:
        return None
    try:
        return {"player_id": str(pid), "role": str(role), "residual": float(res)}
    except (TypeError, ValueError):
        return None


def load_residuals_from_path(path: str | Path) -> ResidualLoadReport:
    path = Path(path)
    warnings: list[str] = []
    if not path.exists():
        return ResidualLoadReport([], f"missing:{path}", 0, 0, [f"file not found: {path}"])

    rows: list[dict] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "residuals" in data:
            data = data["residuals"]
        if not isinstance(data, list):
            return ResidualLoadReport([], str(path), 0, 0, ["JSON root is not a list"])
        for raw in data:
            if isinstance(raw, dict):
                norm = _normalize_row(raw)
                if norm:
                    rows.append(norm)
    elif path.suffix.lower() in (".parquet", ".pq"):
        try:
            import polars as pl
            df = pl.read_parquet(path)
            for raw in df.to_dicts():
                norm = _normalize_row(raw)
                if norm:
                    rows.append(norm)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"parquet read failed: {exc}")
    else:
        warnings.append(f"unsupported residual file type: {path.suffix}")

    players = {r["player_id"] for r in rows}
    roles = {r["role"] for r in rows}
    return ResidualLoadReport(rows, str(path), len(players), len(roles), warnings)


def load_residuals_from_artifacts(artifacts_dir: str | Path | None) -> ResidualLoadReport:
    """Try common artifact locations under *artifacts_dir*."""
    if not artifacts_dir:
        return ResidualLoadReport([], "none", 0, 0, ["artifacts_dir not set"])
    root = Path(artifacts_dir)
    candidates = [
        root / "residuals.json",
        root / "residuals.parquet",
        root / "evaluation" / "residuals.json",
        root / "evaluation" / "residuals.parquet",
        root / "mc" / "residuals.json",
    ]
    for c in candidates:
        if c.exists():
            report = load_residuals_from_path(c)
            if report.residuals:
                return report
    return ResidualLoadReport(
        [],
        "not_found",
        0,
        0,
        [f"no residual file under {root}; tried {[str(c.name) for c in candidates]}"],
    )



def load_residuals_preferring_r2(
    artifacts_dir: str | Path | None = None,
    *,
    r2_endpoint_url: str | None = None,
    r2_access_key_id: str | None = None,
    r2_secret_access_key: str | None = None,
    r2_bucket_name: str | None = None,
) -> ResidualLoadReport:
    """Load residuals from local disk, then pull ``residuals.json`` from R2 if missing.

    Order:
    1. Local candidates under artifacts_dir (same as ``load_residuals_from_artifacts``)
    2. ArtifactStore.load_json("residuals.json") — downloads from R2 into local cache
    3. Empty report with warnings
    """
    import os

    art = artifacts_dir or os.environ.get("API_ARTIFACTS_DIR") or os.environ.get("ARTIFACTS_DIR")
    local = load_residuals_from_artifacts(art)
    if local.residuals:
        return local

    # Attempt R2 via ArtifactStore
    try:
        from ml.storage.artifact_store import ArtifactStore, R2Config
    except Exception as exc:  # noqa: BLE001
        return ResidualLoadReport(
            [],
            "r2_unavailable",
            0,
            0,
            local.warnings + [f"ArtifactStore import failed: {exc}"],
        )

    root = Path(art) if art else Path("artifacts")
    root.mkdir(parents=True, exist_ok=True)

    endpoint = r2_endpoint_url or os.environ.get("API_R2_ENDPOINT_URL") or os.environ.get("ML_R2_ENDPOINT_URL")
    key_id = r2_access_key_id or os.environ.get("API_R2_ACCESS_KEY_ID") or os.environ.get("ML_R2_ACCESS_KEY_ID")
    secret = r2_secret_access_key or os.environ.get("API_R2_SECRET_ACCESS_KEY") or os.environ.get("ML_R2_SECRET_ACCESS_KEY")
    bucket = r2_bucket_name or os.environ.get("API_R2_BUCKET_NAME") or os.environ.get("ML_R2_BUCKET_NAME")

    r2_cfg = None
    if endpoint and key_id and secret and bucket:
        r2_cfg = R2Config(
            endpoint_url=endpoint,
            access_key_id=key_id,
            secret_access_key=secret,
            bucket_name=bucket,
        )

    store = ArtifactStore(local_dir=root, r2_config=r2_cfg)
    data = store.load_json("residuals.json")
    if data is None:
        return ResidualLoadReport(
            [],
            "not_found_local_or_r2",
            0,
            0,
            local.warnings + ["residuals.json not found locally or on R2"],
        )

    # Reuse path loader after download (file now on disk) or parse in-memory
    local_path = root / "residuals.json"
    if local_path.exists():
        report = load_residuals_from_path(local_path)
        if report.residuals:
            report.source = f"r2_or_local:{local_path}"
            report.warnings = local.warnings + report.warnings
            return report

    # In-memory parse if load_json returned data without writing (shouldn't happen)
    rows: list[dict] = []
    warnings = list(local.warnings)
    payload = data
    if isinstance(data, dict) and "residuals" in data:
        payload = data["residuals"]
    if isinstance(payload, list):
        for raw in payload:
            if isinstance(raw, dict):
                norm = _normalize_row(raw)
                if norm:
                    rows.append(norm)
    else:
        warnings.append("residuals.json payload is not a list")
    players = {r["player_id"] for r in rows}
    roles = {r["role"] for r in rows}
    return ResidualLoadReport(rows, "r2:residuals.json", len(players), len(roles), warnings)



def merge_with_prediction_std(
    residuals: Sequence[dict],
    pool: Sequence[Any],
    *,
    random_seed: int = 42,
    n_synthetic: int = 40,
) -> list[dict]:
    """Augment sparse residuals with synthetic draws from Player.prediction_std."""
    import numpy as np

    existing_ids = {r["player_id"] for r in residuals}
    out = list(residuals)
    rng = np.random.default_rng(random_seed)
    for p in pool:
        pid = getattr(p, "player_id", None)
        std = getattr(p, "prediction_std", None)
        if pid is None or pid in existing_ids:
            continue
        if std is None or std <= 0:
            continue
        role = getattr(p, "role", "C")
        for val in rng.normal(0.0, float(std), size=n_synthetic):
            out.append({"player_id": pid, "role": role, "residual": float(val)})
    return out
