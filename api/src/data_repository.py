"""DataRepository: decouples ML artifact I/O from API route logic.

Reads ``results_latest.json`` (and optionally ``next_season_predictions.json``)
produced by the ML trainer pipeline.  A Redis cache layer avoids repeated disk
access for the same static file.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Cache key used in Redis for the latest ML result artifact.
_CACHE_KEY = "ml:results_latest"
_NEXT_CACHE_KEY = "ml:next_season_predictions"


class DataRepository:
    """Thin async repository over serialised ML artifacts.

    Args:
        artifacts_dir: Filesystem path to the directory that contains
            ``results_latest.json`` and companion prediction files.
        redis_client: An optional ``redis.asyncio`` client.  When *None*,
            caching is disabled and every read goes directly to disk.
        cache_ttl: TTL in seconds for Redis cache entries (default 1 h).
    """

    def __init__(
        self,
        artifacts_dir: Path,
        redis_client: Any | None = None,
        cache_ttl: int = 3600,
    ) -> None:
        self._dir = artifacts_dir
        self._redis = redis_client
        self._ttl = cache_ttl

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _read_json(self, path: Path) -> dict:
        """Read a JSON file from disk in a thread-pool executor."""
        try:
            import orjson  # type: ignore[import]

            def _load() -> dict:
                return orjson.loads(path.read_bytes())
        except ImportError:
            import json

            def _load() -> dict:  # type: ignore[misc]
                return json.loads(path.read_text(encoding="utf-8"))

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load)

    async def _cached(self, key: str, path: Path) -> dict:
        """Return JSON data from Redis cache or fall back to disk."""
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                if raw is not None:
                    try:
                        import orjson  # type: ignore[import]

                        return orjson.loads(raw)
                    except ImportError:
                        import json

                        return json.loads(raw)
            except Exception:
                log.warning("Redis read failed for key=%s; falling back to disk", key)

        data = await self._read_json(path)

        if self._redis is not None:
            try:
                try:
                    import orjson  # type: ignore[import]

                    payload = orjson.dumps(data)
                except ImportError:
                    import json

                    payload = json.dumps(data)
                await self._redis.setex(key, self._ttl, payload)
            except Exception:
                log.warning("Redis write failed for key=%s", key)

        return data

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_latest_results(self) -> dict:
        """Load the full ``results_latest.json`` artifact (cached)."""
        path = self._dir / "results_latest.json"
        if not path.exists():
            raise FileNotFoundError(f"No ML artifact found at {path}")
        return await self._cached(_CACHE_KEY, path)

    async def get_predictions(self) -> list[dict]:
        data = await self.get_latest_results()
        return data.get("predictions", [])

    async def get_model_comparison(self) -> list[dict]:
        data = await self.get_latest_results()
        return data.get("model_comparison", [])

    async def get_run_metadata(self) -> dict:
        data = await self.get_latest_results()
        return {
            "run_id": data.get("run_id", ""),
            "best_model": data.get("best_model", ""),
            "role_partitioned": data.get("role_partitioned", False),
        }

    async def get_next_season_predictions(self) -> list[dict]:
        """Return next-season predictions from the companion file if present."""
        next_path = self._dir / "next_season_predictions.json"
        if next_path.exists():
            data = await self._cached(_NEXT_CACHE_KEY, next_path)
            # File may itself be a list or a dict with a list inside.
            return data if isinstance(data, list) else data.get("next_season_predictions", [])
        # Fall back to embedded key in the main artifact.
        data = await self.get_latest_results()
        return data.get("next_season_predictions", [])

    async def get_player_clusters(self) -> list[dict]:
        data = await self.get_latest_results()
        return data.get("player_clusters", [])

    async def get_low_cost_recommendations(
        self,
        top_player_id: Optional[int] = None,
    ) -> list[dict]:
        data = await self.get_latest_results()
        recs: list[dict] = data.get("low_cost_recommendations", [])
        if top_player_id is not None:
            recs = [r for r in recs if r.get("top_player_id") == top_player_id]
        return recs

    async def get_clustering_stats(self) -> dict:
        data = await self.get_latest_results()
        return data.get("clustering_stats", {})

    async def invalidate_cache(self) -> None:
        """Evict Redis cache entries so the next request re-reads from disk."""
        if self._redis is not None:
            try:
                await self._redis.delete(_CACHE_KEY, _NEXT_CACHE_KEY)
                log.info("ML result cache invalidated")
            except Exception:
                log.warning("Cache invalidation failed")
