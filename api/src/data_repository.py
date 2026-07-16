"""DataRepository: decouples ML artifact I/O from API route logic.

Reads ``results_latest.json`` (and optionally ``next_season_predictions.json``)
produced by the ML trainer pipeline.  A Redis cache layer avoids repeated disk
access for the same static file.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Final, Optional

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import PlayerIdMap, PlayerQuotation

log = logging.getLogger(__name__)

# Cache key used in Redis for the latest ML result artifact.
_CACHE_KEY = "ml:results_latest"
_NEXT_CACHE_KEY = "ml:next_season_predictions"


# Mapping from canonical ML role names (English) to optimizer role codes
# (Fantacalcio classic). Kept module-level so it is reused across requests.
_CANONICAL_ROLE_TO_OPTIMIZER: Final[dict[str, str]] = {
    "GK": "P",
    "DEF": "D",
    "MID": "C",
    "FWD": "A",
}

# Normalisation of FotMob / predictions team names to Fantacalcio canonical
# names (used by ``OptimizationConfig.big_teams`` default). Anything not in
# the map is left untouched.
_TEAM_NAME_NORMALISATION: Final[dict[str, str]] = {
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "FC Internazionale": "Inter",
    "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli",
    "AC Milan": "Milan",
}


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

    async def get_var_results(self) -> list[dict]:
        """Return VAR/ESV records from the artifact if present."""
        data = await self.get_latest_results()
        return data.get("var_results", [])

    async def invalidate_cache(self) -> None:
        """Evict Redis cache entries so the next request re-reads from disk."""
        if self._redis is not None:
            try:
                await self._redis.delete(_CACHE_KEY, _NEXT_CACHE_KEY)
                log.info("ML result cache invalidated")
            except Exception:
                log.warning("Cache invalidation failed")

    # ── Quotations (DB-backed) ────────────────────────────────────────────────

    @staticmethod
    def _build_quotation_filters(
        season_start: Optional[int],
        role: Optional[str],
        team: Optional[str],
        player_fotmob_id: Optional[int],
        min_qt_a: Optional[int],
        max_qt_a: Optional[int],
    ) -> list:
        """Compose WHERE clauses for ``list_quotations``."""
        filters = []
        if season_start is not None:
            filters.append(PlayerQuotation.season_start == season_start)
        if role is not None:
            filters.append(PlayerQuotation.role == role)
        if team is not None:
            filters.append(PlayerQuotation.team == team)
        if min_qt_a is not None:
            filters.append(PlayerQuotation.qt_a >= min_qt_a)
        if max_qt_a is not None:
            filters.append(PlayerQuotation.qt_a <= max_qt_a)
        if player_fotmob_id is not None:
            filters.append(PlayerIdMap.player_fotmob_id == player_fotmob_id)
        return filters

    async def list_quotations(
        self,
        db: AsyncSession,
        season_start: Optional[int] = None,
        role: Optional[str] = None,
        team: Optional[str] = None,
        player_fotmob_id: Optional[int] = None,
        min_qt_a: Optional[int] = None,
        max_qt_a: Optional[int] = None,
        page: int = 1,
        size: int = 50,
    ) -> tuple[list[dict], int]:
        """Return ``(rows, total)`` from ``player_quotations`` joined to id-map.

        Each row is a dict with both quotation and mapping fields (left join,
        so ``player_fotmob_id``/``match_method`` may be ``None``).
        """
        filters = self._build_quotation_filters(
            season_start, role, team, player_fotmob_id, min_qt_a, max_qt_a
        )

        # Outer join quotation ↔ id-map on (fantacalcio_id, season_start).
        join_cond = and_(
            PlayerQuotation.fantacalcio_id == PlayerIdMap.fantacalcio_id,
            PlayerQuotation.season_start == PlayerIdMap.season_start,
        )

        base_cols = [
            PlayerQuotation.id,
            PlayerQuotation.fantacalcio_id,
            PlayerQuotation.season_start,
            PlayerQuotation.role,
            PlayerQuotation.team,
            PlayerQuotation.player_name,
            PlayerQuotation.qt_a,
            PlayerQuotation.qt_i,
            PlayerQuotation.diff_val,
            PlayerQuotation.qt_a_m,
            PlayerQuotation.qt_i_m,
            PlayerQuotation.diff_val_m,
            PlayerQuotation.fvm,
            PlayerQuotation.fvm_m,
            PlayerQuotation.source,
            PlayerQuotation.imported_at,
            PlayerIdMap.player_fotmob_id,
            PlayerIdMap.name_fotmob,
            PlayerIdMap.team_fotmob,
            PlayerIdMap.match_method,
            PlayerIdMap.confidence,
        ]

        count_stmt = select(func.count()).select_from(PlayerQuotation)
        if filters:
            if player_fotmob_id is not None:
                # Cross-table filter → must include the join.
                count_stmt = count_stmt.join(PlayerIdMap, join_cond)
            count_stmt = count_stmt.where(*filters)
        total = (await db.execute(count_stmt)).scalar_one()

        stmt = select(*base_cols).select_from(PlayerQuotation)
        stmt = stmt.join(PlayerIdMap, join_cond, isouter=True)
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.order_by(
            PlayerQuotation.season_start.desc(),
            PlayerQuotation.role,
            PlayerQuotation.qt_a.desc(),
        )
        stmt = stmt.offset(max(0, (page - 1) * size)).limit(max(1, size))

        result = await db.execute(stmt)
        rows = [dict(r._mapping) for r in result]
        # Normalise enum → str and confidence → float.
        for r in rows:
            if r.get("match_method") is not None and not isinstance(r["match_method"], str):
                r["match_method"] = r["match_method"].value
            if r.get("confidence") is not None and not isinstance(r["confidence"], (int, float)):
                r["confidence"] = float(r["confidence"])
            if r.get("imported_at") is not None and not isinstance(r["imported_at"], str):
                r["imported_at"] = r["imported_at"].isoformat()
        return rows, total

    async def get_quotation_seasons(self, db: AsyncSession) -> list[int]:
        """Distinct ``season_start`` values present in ``player_quotations``."""
        stmt = (
            select(PlayerQuotation.season_start)
            .distinct()
            .order_by(PlayerQuotation.season_start.desc())
        )
        result = await db.execute(stmt)
        return [row[0] for row in result]

    async def get_quotation_stats(self, db: AsyncSession) -> dict:
        """Aggregate counts, role+season means, and id-mapping coverage."""
        # total + n_teams + seasons
        base_stmt = select(func.count()).select_from(PlayerQuotation)
        total = (await db.execute(base_stmt)).scalar_one()

        seasons = await self.get_quotation_seasons(db)

        n_teams_stmt = select(func.count(func.distinct(PlayerQuotation.team)))
        n_teams = (await db.execute(n_teams_stmt)).scalar_one()

        # Per role+season aggregates
        per_role = (
            select(
                PlayerQuotation.season_start,
                PlayerQuotation.role,
                func.count().label("n"),
                func.avg(PlayerQuotation.qt_a).label("avg_qt_a"),
                func.avg(PlayerQuotation.qt_i).label("avg_qt_i"),
                func.percentile_cont(0.5)
                .within_group(PlayerQuotation.qt_a.asc())
                .label("median_qt_a"),
                func.min(PlayerQuotation.qt_a).label("min_qt_a"),
                func.max(PlayerQuotation.qt_a).label("max_qt_a"),
                func.avg(PlayerQuotation.fvm).label("avg_fvm"),
            )
            .group_by(PlayerQuotation.season_start, PlayerQuotation.role)
            .order_by(PlayerQuotation.season_start.desc(), PlayerQuotation.role)
        )
        rows = (await db.execute(per_role)).all()
        by_season_role = [
            {
                "season_start": r.season_start,
                "role": r.role,
                "n_players": r.n,
                "avg_qt_a": float(r.avg_qt_a) if r.avg_qt_a is not None else 0.0,
                "avg_qt_i": float(r.avg_qt_i) if r.avg_qt_i is not None else 0.0,
                "median_qt_a": float(r.median_qt_a) if r.median_qt_a is not None else 0.0,
                "min_qt_a": int(r.min_qt_a) if r.min_qt_a is not None else 0,
                "max_qt_a": int(r.max_qt_a) if r.max_qt_a is not None else 0,
                "avg_fvm": float(r.avg_fvm) if r.avg_fvm is not None else None,
            }
            for r in rows
        ]

        # id-mapping coverage (count by method) — left-joined to quotations so
        # we only see methods for rows that actually have a quotation.
        cov_stmt = (
            select(PlayerIdMap.match_method, func.count().label("n"))
            .select_from(PlayerIdMap)
            .join(
                PlayerQuotation,
                and_(
                    PlayerIdMap.fantacalcio_id == PlayerQuotation.fantacalcio_id,
                    PlayerIdMap.season_start == PlayerQuotation.season_start,
                ),
            )
            .group_by(PlayerIdMap.match_method)
        )
        cov_rows = (await db.execute(cov_stmt)).all()
        coverage: dict[str, int] = {}
        for r in cov_rows:
            method = r.match_method.value if not isinstance(r.match_method, str) else r.match_method
            coverage[method] = int(r.n)

        return {
            "total_quotations": int(total),
            "seasons": seasons,
            "by_season_role": by_season_role,
            "n_teams": int(n_teams),
            "coverage": coverage,
        }

    # ── ID-mapping (DB-backed) ────────────────────────────────────────────────

    async def list_id_mappings(
        self,
        db: AsyncSession,
        season_start: Optional[int] = None,
        match_method: Optional[str] = None,
        canonical_role: Optional[str] = None,
        matched_only: bool = False,
        page: int = 1,
        size: int = 50,
    ) -> tuple[list[dict], int]:
        """Paginated listing of ``player_id_map`` rows."""
        filters = []
        if season_start is not None:
            filters.append(PlayerIdMap.season_start == season_start)
        if match_method is not None:
            filters.append(PlayerIdMap.match_method == match_method)
        if canonical_role is not None:
            filters.append(PlayerIdMap.canonical_role == canonical_role)
        if matched_only:
            filters.append(PlayerIdMap.player_fotmob_id.is_not(None))

        count_stmt = select(func.count()).select_from(PlayerIdMap)
        if filters:
            count_stmt = count_stmt.where(*filters)
        total = (await db.execute(count_stmt)).scalar_one()

        stmt = select(PlayerIdMap)
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.order_by(
            PlayerIdMap.season_start.desc(),
            PlayerIdMap.fantacalcio_id,
        )
        stmt = stmt.offset(max(0, (page - 1) * size)).limit(max(1, size))

        result = await db.execute(stmt)
        rows = [m.to_dict() for m in result.scalars().all()]
        return rows, total

    async def get_id_mapping(
        self,
        db: AsyncSession,
        fantacalcio_id: int,
        season_start: int,
    ) -> Optional[dict]:
        stmt = select(PlayerIdMap).where(
            and_(
                PlayerIdMap.fantacalcio_id == fantacalcio_id,
                PlayerIdMap.season_start == season_start,
            )
        )
        result = await db.execute(stmt)
        m = result.scalar_one_or_none()
        return m.to_dict() if m is not None else None

    async def get_id_mapping_stats(self, db: AsyncSession) -> dict:
        """Match rate + breakdowns by season and by method."""
        total_stmt = select(func.count()).select_from(PlayerIdMap)
        total = (await db.execute(total_stmt)).scalar_one()

        matched_stmt = select(func.count()).select_from(PlayerIdMap).where(
            PlayerIdMap.player_fotmob_id.is_not(None)
        )
        matched = (await db.execute(matched_stmt)).scalar_one()
        unmatched = int(total) - int(matched)

        # by_method
        by_method_rows = (
            await db.execute(
                select(PlayerIdMap.match_method, func.count().label("n")).group_by(
                    PlayerIdMap.match_method
                )
            )
        ).all()
        by_method: dict[str, int] = {}
        for r in by_method_rows:
            method = r.match_method.value if not isinstance(r.match_method, str) else r.match_method
            by_method[method] = int(r.n)

        # by_season (nested)
        by_season_rows = (
            await db.execute(
                select(
                    PlayerIdMap.season_start,
                    PlayerIdMap.match_method,
                    func.count().label("n"),
                ).group_by(PlayerIdMap.season_start, PlayerIdMap.match_method)
            )
        ).all()
        by_season: dict[str, dict[str, int]] = {}
        for r in by_season_rows:
            method = r.match_method.value if not isinstance(r.match_method, str) else r.match_method
            by_season.setdefault(str(r.season_start), {})[method] = int(r.n)

        return {
            "total": int(total),
            "matched": int(matched),
            "unmatched": int(unmatched),
            "match_rate": (int(matched) / int(total)) if int(total) else 0.0,
            "by_season": by_season,
            "by_method": by_method,
        }

    async def update_id_mapping(
        self,
        db: AsyncSession,
        fantacalcio_id: int,
        season_start: int,
        *,
        player_fotmob_id: Optional[int] = None,
        name_fotmob: Optional[str] = None,
        team_fotmob: Optional[str] = None,
        canonical_role: Optional[str] = None,
    ) -> Optional[dict]:
        """Manually update a single row in ``player_id_map``.

        Sets ``match_method='manual'`` and ``confidence=1.0`` so the
        override is clearly distinguishable from automatic matches.

        Args:
            db: DB session.
            fantacalcio_id: Fantacalcio ID to update.
            season_start: Season start year.
            player_fotmob_id: FotMob ID to assign (``None`` = leave as-is,
                ``-1`` = explicitly clear/set unmatched).
            name_fotmob: FotMob player name (informational).
            team_fotmob: FotMob team name override.
            canonical_role: Canonical role override (GK/DEF/MID/FWD).

        Returns:
            The updated row dict, or ``None`` if the row was not found.
        """
        from datetime import datetime, timezone

        stmt = select(PlayerIdMap).where(
            and_(
                PlayerIdMap.fantacalcio_id == fantacalcio_id,
                PlayerIdMap.season_start == season_start,
            )
        )
        result = await db.execute(stmt)
        mapping = result.scalar_one_or_none()
        if mapping is None:
            return None

        # player_fotmob_id: -1 means "explicitly unmatched"
        if player_fotmob_id is not None:
            mapping.player_fotmob_id = None if player_fotmob_id == -1 else player_fotmob_id
        if name_fotmob is not None:
            mapping.name_fotmob = name_fotmob
        if team_fotmob is not None:
            mapping.team_fotmob = team_fotmob
        if canonical_role is not None:
            mapping.canonical_role = canonical_role

        mapping.match_method = MatchMethodEnum.MANUAL
        mapping.confidence = 1.0
        mapping.updated_at = datetime.now(tz=timezone.utc)

        await db.commit()
        await db.refresh(mapping)
        return mapping.to_dict()

    async def get_player_fotmob_history(
        self,
        db: AsyncSession,
        player_fotmob_id: int,
    ) -> list[dict]:
        """Quotation history for a single FotMob player across all seasons."""
        stmt = (
            select(PlayerQuotation, PlayerIdMap)
            .join(
                PlayerIdMap,
                and_(
                    PlayerIdMap.fantacalcio_id == PlayerQuotation.fantacalcio_id,
                    PlayerIdMap.season_start == PlayerQuotation.season_start,
                ),
            )
            .where(PlayerIdMap.player_fotmob_id == player_fotmob_id)
            .order_by(PlayerQuotation.season_start.asc())
        )
        result = await db.execute(stmt)
        out: list[dict] = []
        for pq, pim in result.all():
            row = pq.to_dict()
            row["player_fotmob_id"] = pim.player_fotmob_id
            row["name_fotmob"] = pim.name_fotmob
            row["team_fotmob"] = pim.team_fotmob
            row["match_method"] = (
                pim.match_method.value if not isinstance(pim.match_method, str) else pim.match_method
            )
            row["confidence"] = float(pim.confidence) if pim.confidence is not None else None
            out.append(row)
        return out

    # ── Optimizer pool (DB + ML artifact join) ────────────────────────────────

    @staticmethod
    def _normalise_team(name: str) -> str:
        """Return the Fantacalcio canonical team name.

        Anything missing from :data:`_TEAM_NAME_NORMALISATION` is returned
        unchanged so we never silently mislabel an unknown team.
        """
        return _TEAM_NAME_NORMALISATION.get(name, name)

    @staticmethod
    def _to_optimizer_role(canonical_role: str) -> Optional[str]:
        """Map a ML canonical role to an optimizer role code (or ``None``)."""
        return _CANONICAL_ROLE_TO_OPTIMIZER.get(canonical_role)

    async def get_player_pool(
        self,
        db: AsyncSession,
        *,
        season_start: int,
        min_qt_a: int = 1,
    ) -> list[dict]:
        """Return optimizer-ready player records joined with ML predictions.

        Output schema (one dict per player, suitable for the optimizer
        ``Player`` dataclass):

        * ``player_id`` — stable identifier (``"fc-{fantacalcio_id}"`` or
          ``"fm-{player_fotmob_id}"`` when the id-map is missing the entry).
        * ``name`` — best-effort name (``name_fotmob`` if available,
          otherwise ``player_name``).
        * ``role`` — optimizer role code (``P``/``D``/``C``/``A``); rows
          whose role is not recognised are dropped.
        * ``real_team`` — Fantacalcio canonical team name.
        * ``cost`` — current quotation (``qt_a``).
        * ``projected_score`` — ML ``predicted_fantavoto`` if available,
          otherwise ``fantavoto_medio`` (historical mean) as fallback.

        The list is *not* deduplicated; the optimizer orchestrator is
        responsible for that. The repository guarantees deterministic
        ordering: by role, then by ``qt_a`` desc, then by ``player_id`` for
        reproducibility across requests.
        """
        # 1) Quotations joined to id-map (left join — players without
        #    fotmob mapping still surface, with player_id derived from
        #    fantacalcio_id).
        join_cond = and_(
            PlayerQuotation.fantacalcio_id == PlayerIdMap.fantacalcio_id,
            PlayerQuotation.season_start == PlayerIdMap.season_start,
        )
        stmt = (
            select(PlayerQuotation, PlayerIdMap)
            .select_from(PlayerQuotation)
            .join(PlayerIdMap, join_cond, isouter=True)
            .where(PlayerQuotation.season_start == season_start)
            .where(PlayerQuotation.qt_a >= min_qt_a)
        )
        rows = (await db.execute(stmt)).all()

        # 2) ML predictions indexed by player_fotmob_id.
        predictions_by_id: dict[int, dict] = {}
        try:
            preds = await self.get_predictions()
        except FileNotFoundError:
            preds = []
        for p in preds:
            pid = p.get("player_fotmob_id")
            if pid is not None:
                predictions_by_id[int(pid)] = p

        # 3) Project into the optimizer-friendly shape.
        pool: list[dict] = []
        for pq, pim in rows:
            optimizer_role = self._to_optimizer_role(pq.role)
            if optimizer_role is None:
                # Unknown / unsupported role (e.g. outdated enum) — skip.
                continue

            cost = int(pq.qt_a) if pq.qt_a is not None else 0
            if cost <= 0:
                continue

            # Best-effort player_id and name.
            if pim is not None and pim.player_fotmob_id is not None:
                player_id = f"fm-{pim.player_fotmob_id}"
                name = pim.name_fotmob or pq.player_name
            else:
                player_id = f"fc-{pq.fantacalcio_id}"
                name = pq.player_name

            # ML prediction: prefer predicted_fantavoto, fallback to mean.
            predicted_score: Optional[float] = None
            if pim is not None and pim.player_fotmob_id is not None:
                pred = predictions_by_id.get(int(pim.player_fotmob_id))
                if pred is not None:
                    val = pred.get("predicted_fantavoto")
                    if isinstance(val, (int, float)):
                        predicted_score = float(val)
                    elif isinstance(pred.get("fantavoto_medio"), (int, float)):
                        predicted_score = float(pred["fantavoto_medio"])
            if predicted_score is None and isinstance(pq.fvm, (int, float)):
                # Historical fantavoto medio from the quotation itself.
                predicted_score = float(pq.fvm)
            if predicted_score is None or predicted_score <= 0.0:
                # Without a meaningful projection the optimizer cannot
                # rank this player — skip.
                continue

            # Best-effort team name (id-map wins when present).
            if pim is not None and pim.team_fotmob:
                real_team = self._normalise_team(pim.team_fotmob)
            else:
                real_team = self._normalise_team(pq.team or "")

            pool.append(
                {
                    "player_id": player_id,
                    "name": name,
                    "role": optimizer_role,
                    "real_team": real_team,
                    "cost": cost,
                    "projected_score": predicted_score,
                }
            )

        # Deterministic ordering for reproducibility.
        pool.sort(key=lambda r: (r["role"], -r["cost"], r["player_id"]))
        return pool
