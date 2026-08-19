"""Process-local TTL store for RosterContext objects.

Runtime-only: no Redis, no database. Contexts expire after ``ttl_seconds``
and are also evicted when the process restarts. Suitable for single-instance
deployments; for multi-instance the client must re-upload or the context
payload can be returned fully in the import response (stateless mode).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ml.roster_import.context import RosterContext

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600  # 1 hour


@dataclass
class _Entry:
    context: RosterContext
    expires_at: float


class RosterContextStore:
    """Thread-safe in-process store with TTL eviction."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = max(60, ttl_seconds)
        self._lock = threading.RLock()
        self._data: dict[str, _Entry] = {}

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def put(self, context: RosterContext) -> str:
        """Store ``context`` and return its ``context_id``."""
        with self._lock:
            self._purge_expired()
            self._data[context.context_id] = _Entry(
                context=context,
                expires_at=time.monotonic() + self._ttl,
            )
            log.debug(
                "RosterContext stored id=%s teams=%d ttl=%ds",
                context.context_id,
                context.quality.total_players,
                self._ttl,
            )
            return context.context_id

    def get(self, context_id: str) -> Optional[RosterContext]:
        with self._lock:
            entry = self._data.get(context_id)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._data[context_id]
                log.debug("RosterContext expired id=%s", context_id)
                return None
            return entry.context

    def update(self, context: RosterContext) -> None:
        """Replace an existing context (e.g. after claim user team)."""
        with self._lock:
            if context.context_id not in self._data:
                # treat as put
                self.put(context)
                return
            self._data[context.context_id] = _Entry(
                context=context,
                expires_at=time.monotonic() + self._ttl,
            )

    def delete(self, context_id: str) -> bool:
        with self._lock:
            return self._data.pop(context_id, None) is not None

    def size(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._data)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, e in self._data.items() if e.expires_at < now]
        for k in expired:
            del self._data[k]
        if expired:
            log.debug("Purged %d expired RosterContext entries", len(expired))


# Module-level singleton used by the API process.
_default_store: RosterContextStore | None = None
_store_lock = threading.Lock()


def get_default_store(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> RosterContextStore:
    global _default_store
    with _store_lock:
        if _default_store is None:
            _default_store = RosterContextStore(ttl_seconds=ttl_seconds)
        return _default_store


def reset_default_store() -> None:
    """Test helper: drop the singleton."""
    global _default_store
    with _store_lock:
        _default_store = None
