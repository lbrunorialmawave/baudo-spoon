"""End-to-end season onboarding orchestrator.

The command intentionally delegates each existing pipeline stage to its current
implementation instead of duplicating MANTRA/ML business logic:

    quotations import -> foreign stats -> MANTRA -> ML workflow -> hybrid

Training is dispatched through the existing admin API because the API/ML
containers are separate deployables.  The training status endpoint is polled
until the newly-triggered run completes, then the R2-backed artifact store is
used to hydrate ``results_latest.json`` locally before the hybrid artifact is
regenerated.
"""

from __future__ import annotations

import argparse
import hmac
import hashlib
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import sqlalchemy as sa

log = logging.getLogger("season_refresh")


@dataclass
class StepResult:
    name: str
    status: str
    detail: dict[str, Any]


class RefreshError(RuntimeError):
    pass


def _sync_db_url(url: str) -> str:
    return (
        url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgres+asyncpg://", "postgres+psycopg2://")
        .replace("?ssl=", "?sslmode=")
        .replace("&ssl=", "&sslmode=")
    )


def _run(cmd: list[str], name: str) -> StepResult:
    log.info("[%s] %s", name, " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        log.info("[%s] stdout: %s", name, proc.stdout[-2000:])
    if proc.stderr:
        log.info("[%s] stderr: %s", name, proc.stderr[-2000:])
    if proc.returncode != 0:
        raise RefreshError(f"{name} failed (exit={proc.returncode})")
    return StepResult(name, "ok", {"returncode": proc.returncode})


def _coherence_check(engine: sa.Engine, quotations_dir: Path) -> dict[str, Any]:
    files = sorted(quotations_dir.glob("Quotazioni_Fantacalcio_Stagione_*.xlsx"))
    latest_db = engine.connect().execute(sa.text("SELECT MAX(season_start) FROM player_quotations")).scalar()
    latest_file = None
    if files:
        import re
        seasons = [int(m.group(1)) for f in files if (m := re.search(r"Stagione_(\d{4})_", f.name))]
        latest_file = max(seasons) if seasons else None
    if latest_file is None:
        raise RefreshError(f"No quotation XLSX found in {quotations_dir}")
    return {
        "latest_file_season": latest_file,
        "latest_db_season": int(latest_db) if latest_db is not None else None,
        "db_has_quotations": latest_db is not None,
        "season_aligned": latest_db is None or int(latest_db) == latest_file,
        "quotation_files": len(files),
    }


def _foreign_stats(engine: sa.Engine) -> dict[str, Any]:
    latest = engine.connect().execute(sa.text("SELECT MAX(season_start) FROM player_quotations")).scalar()
    if latest is None:
        return {"status": "skipped", "reason": "no_quotations", "candidates": 0}
    sql = sa.text("""
        SELECT DISTINCT pim.player_fotmob_id, pq.player_name
        FROM player_quotations pq
        JOIN player_id_map pim
          ON pim.fantacalcio_id = pq.fantacalcio_id
         AND pim.season_start = pq.season_start
        LEFT JOIN player_season_aggregates pss_cur
          ON pss_cur.fantacalcio_id = pim.player_fotmob_id::bigint
         AND pss_cur.season_start = pq.season_start
        LEFT JOIN player_season_aggregates pss_prev
          ON pss_prev.fantacalcio_id = pim.player_fotmob_id::bigint
         AND pss_prev.season_start = pq.season_start - 1
        LEFT JOIN player_latest_stats_any_league pss_any
          ON pss_any.fantacalcio_id = pim.player_fotmob_id::bigint
        WHERE pq.season_start = :season_start
          AND pim.player_fotmob_id IS NOT NULL
          AND pss_cur.fantacalcio_id IS NULL
          AND pss_prev.fantacalcio_id IS NULL
          AND pss_any.fantacalcio_id IS NULL
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"season_start": int(latest)}).mappings().all()
    candidates = {int(r["player_fotmob_id"]): r["player_name"] for r in rows}
    if not candidates:
        return {"status": "ok", "season_start": int(latest), "candidates": 0, "fetched": 0, "persisted": 0}
    from scraper.src.player_career_scraper import fetch_and_persist_players
    fetched, persisted = fetch_and_persist_players(candidates, _sync_db_url(os.environ["ML_DATABASE_URL"]))
    return {
        "status": "ok",
        "season_start": int(latest),
        "candidates": len(candidates),
        "fetched": fetched,
        "persisted": persisted,
        "unresolved": len(candidates) - fetched,
    }


def _signing_secret() -> str:
    # Same shared secret as the server's API_API_KEY_SECRET (settings.api_key_secret).
    # Read here under either name so existing GH Actions secrets keep working
    # without a rename.
    secret = os.environ.get("SEASON_REFRESH_API_KEY") or os.environ.get("API_API_KEY_SECRET")
    if not secret:
        raise RefreshError(
            "No service credential configured — set SEASON_REFRESH_API_KEY "
            "(must match the server's API_API_KEY_SECRET)."
        )
    return secret


def _signed_headers(secret: str, method: str, url_path: str) -> dict[str, str]:
    """HMAC-SHA256 request signature: proves possession of the shared secret
    without putting it on the wire, and binds the signature to method+path+time
    so a captured request can't be replayed against a different endpoint or later.

    url_path MUST be exactly what the server sees as request.url.path (i.e. the
    full path including any router prefix such as /api/v1) — not just the
    relative suffix passed to _api_url — or the signature won't match.
    """
    timestamp = str(int(time.time()))
    message = f"{timestamp}:{method.upper()}:{url_path}".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return {
        "X-Service-Id": "season-refresh",
        "X-API-Timestamp": timestamp,
        "X-API-Signature": signature,
    }


def _api_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _signed_get(secret: str, api_base: str, path: str, **kwargs) -> requests.Response:
    full_url = _api_url(api_base, path)
    url_path = urlparse(full_url).path  # what the server's request.url.path will actually be
    return requests.get(full_url, headers=_signed_headers(secret, "GET", url_path), **kwargs)


def _signed_post(secret: str, api_base: str, path: str, **kwargs) -> requests.Response:
    full_url = _api_url(api_base, path)
    url_path = urlparse(full_url).path
    return requests.post(full_url, headers=_signed_headers(secret, "POST", url_path), **kwargs)


def _trigger_training(api_base: str, poll_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    secret = _signing_secret()
    before = _signed_get(secret, api_base, "/admin/ml/train/status", timeout=20).json()
    triggered_at = time.time()
    resp = _signed_post(secret, api_base, "/admin/ml/train", timeout=30)
    if resp.status_code >= 400:
        raise RefreshError(f"ML training trigger failed: HTTP {resp.status_code}: {resp.text[:500]}")

    deadline = time.time() + timeout_seconds
    last: dict[str, Any] = before
    while time.time() < deadline:
        time.sleep(poll_seconds)
        r = _signed_get(secret, api_base, "/admin/ml/train/status", timeout=20)
        if r.status_code >= 400:
            raise RefreshError(f"ML training status failed: HTTP {r.status_code}: {r.text[:500]}")
        last = r.json()
        log.info("[train] status=%s run_id=%s", last.get("status"), last.get("run_id"))
        if last.get("status") == "failed":
            raise RefreshError(f"ML training failed: {last}")
        updated = last.get("updated_at") or last.get("started_at")
        if last.get("status") == "completed" and updated:
            # If timestamps are unavailable/old, a new run_id is still sufficient.
            if last.get("run_id") != before.get("run_id") or updated != before.get("updated_at") or time.time() - triggered_at > poll_seconds * 2:
                return last
    raise RefreshError(f"Timed out waiting for ML training after {timeout_seconds}s; last status={last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete season onboarding pipeline")
    parser.add_argument("--quotazioni-dir", type=Path, default=Path("quotazioni"))
    parser.add_argument("--db-url", default=os.environ.get("ML_DATABASE_URL"))
    parser.add_argument("--api-url", default=os.environ.get("SEASON_REFRESH_API_URL", "http://localhost:8000/api/v1"))
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--export-unresolved", type=Path, default=Path("artifacts/season_refresh_unresolved.csv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    if not args.db_url:
        log.error("Pass --db-url or set ML_DATABASE_URL")
        return 2

    engine = sa.create_engine(args.db_url, pool_pre_ping=True)
    steps: list[StepResult] = []
    try:
        coherence = _coherence_check(engine, args.quotazioni_dir)
        log.info("[coherence] %s", coherence)
        if not coherence["season_aligned"]:
            log.warning("Quotation file season and DB latest season differ; import will reconcile them.")
        if args.dry_run:
            print({"status": "dry_run", "coherence": coherence})
            return 0

        import_cmd = [sys.executable, "-m", "ml.data.import_quotations", "--quotazioni-dir", str(args.quotazioni_dir), "--db-url", args.db_url, "--export-unresolved", str(args.export_unresolved)]
        if args.overrides:
            import_cmd += ["--overrides", str(args.overrides)]
        steps.append(_run(import_cmd, "import"))

        steps.append(StepResult("foreign-stats", "ok", _foreign_stats(engine)))

        latest = engine.connect().execute(sa.text("SELECT MAX(season_start) FROM player_quotations")).scalar()
        if latest is None:
            raise RefreshError("Import completed but no quotations are present")
        season = int(latest)

        artifacts_dir = Path(os.environ.get("ML_ARTIFACTS_DIR", "artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        from ml.mantra.runner import run_mantra
        mantra_result = run_mantra(engine, season_start=season, output_dir=artifacts_dir)
        steps.append(StepResult("mantra", "ok", {"season_start": season, "n_players": mantra_result["meta"]["n_players"]}))

        train = _trigger_training(args.api_url, args.poll_seconds, args.timeout_seconds)
        steps.append(StepResult("train", "ok", train))

        # Hydrate the freshly-trained artifact from R2, then rebuild the hybrid
        # artifact. This is the same cache-aside mechanism used by the API.
        from ml.storage.artifact_store import ArtifactStore, R2Config
        store = ArtifactStore(local_dir=artifacts_dir, r2_config=R2Config.from_env())
        ml_artifact = store.load_json("results_latest.json")
        if ml_artifact is None:
            raise RefreshError("Training completed but results_latest.json is unavailable locally/R2")

        hybrid_path = artifacts_dir / f"mantra_ibrido_results_{season}.json"
        if hybrid_path.exists():
            hybrid_path.unlink()
            log.info("Invalidated stale hybrid artifact %s", hybrid_path)
        id_map_path = artifacts_dir / "player_id_map.json"
        _run([sys.executable, "-m", "ml.mantra_ibrido.export_id_map", "--db-url", args.db_url, "--output", str(id_map_path)], "export-id-map")
        from ml.mantra_ibrido.runner import run_hybrid_computation
        result = run_hybrid_computation(
            artifacts_dir / f"mantra_results_{season}.json",
            artifacts_dir / "results_latest.json",
            artifacts_dir,
        )
        steps.append(StepResult("hybrid", "ok", {"season_start": season, "n_players": len(result["players"]), "artifact": str(hybrid_path)}))

        print({"status": "ok", "season_start": season, "steps": [asdict(s) for s in steps]})
        return 0
    except Exception as exc:
        log.exception("Season refresh stopped")
        print({"status": "error", "error": str(exc), "completed": [asdict(s) for s in steps]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())