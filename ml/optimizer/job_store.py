"""Job store for async optimizer Monte Carlo runs.

Backends:
* memory (default) — process-local, not multi-replica safe
* redis — shared across API pods when REDIS_URL is available

Env:
  OPTIMIZER_JOB_BACKEND=memory|redis
  OPTIMIZER_JOB_TTL_SECONDS=86400
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "completed", "failed"]

__all__ = ["OptimizeJob", "JobStore", "job_store", "build_job_store"]


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class OptimizeJob:
    job_id: str
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    error: str | None = None
    result: dict[str, Any] | None = None
    monte_carlo_summary: dict[str, Any] | None = None
    request_meta: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result": self.result,
            "monte_carlo_summary": self.monte_carlo_summary,
            "request_meta": self.request_meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeJob":
        return cls(
            job_id=data["job_id"],
            status=data.get("status", "queued"),  # type: ignore[arg-type]
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            error=data.get("error"),
            result=data.get("result"),
            monte_carlo_summary=data.get("monte_carlo_summary"),
            request_meta=data.get("request_meta") or {},
        )


class MemoryJobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, OptimizeJob] = {}

    def create(self, *, request_meta: dict[str, Any] | None = None) -> OptimizeJob:
        job = OptimizeJob(job_id=str(uuid.uuid4()), request_meta=request_meta or {})
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> OptimizeJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "running"
                job.touch()

    def set_completed(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        monte_carlo_summary: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.result = result
                job.monte_carlo_summary = monte_carlo_summary
                job.touch()

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error
                job.touch()


class RedisJobStore:
    """Sync Redis backend (safe to call from worker threads)."""

    def __init__(self, url: str, *, ttl_seconds: int = 86400, prefix: str = "optjob:") -> None:
        import redis  # type: ignore[import]

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def create(self, *, request_meta: dict[str, Any] | None = None) -> OptimizeJob:
        job = OptimizeJob(job_id=str(uuid.uuid4()), request_meta=request_meta or {})
        self._r.set(self._key(job.job_id), json.dumps(job.to_dict()), ex=self._ttl)
        return job

    def get(self, job_id: str) -> OptimizeJob | None:
        raw = self._r.get(self._key(job_id))
        if not raw:
            return None
        return OptimizeJob.from_dict(json.loads(raw))

    def _save(self, job: OptimizeJob) -> None:
        job.touch()
        self._r.set(self._key(job.job_id), json.dumps(job.to_dict()), ex=self._ttl)

    def set_running(self, job_id: str) -> None:
        job = self.get(job_id)
        if job:
            job.status = "running"
            self._save(job)

    def set_completed(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        monte_carlo_summary: dict[str, Any] | None = None,
    ) -> None:
        job = self.get(job_id)
        if job:
            job.status = "completed"
            job.result = result
            job.monte_carlo_summary = monte_carlo_summary
            self._save(job)

    def set_failed(self, job_id: str, error: str) -> None:
        job = self.get(job_id)
        if job:
            job.status = "failed"
            job.error = error
            self._save(job)


# Backwards-compatible alias
JobStore = MemoryJobStore


def build_job_store() -> MemoryJobStore | RedisJobStore:
    backend = os.environ.get("OPTIMIZER_JOB_BACKEND", "memory").lower()
    ttl = int(os.environ.get("OPTIMIZER_JOB_TTL_SECONDS", "86400"))
    if backend == "redis":
        url = os.environ.get("REDIS_URL") or os.environ.get("API_REDIS_URL") or "redis://localhost:6379/0"
        try:
            store = RedisJobStore(url, ttl_seconds=ttl)
            store._r.ping()
            log.info("optimizer job_store backend=redis ttl=%s", ttl)
            return store
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis job store unavailable (%s); falling back to memory", exc)
    return MemoryJobStore()


job_store = build_job_store()
