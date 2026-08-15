"""Deterministic configuration hashing (WS16 of plan.md).

Goal
----
Every persistent artefact that records a *deployment decision* (rollout
state, promotion report, shadow comparison, audit entry) must carry a
**canonical SHA-256** of the configuration that produced it.  This makes
it impossible — by construction — to promote a report generated against
a config that differs from the one actually active in production.

Definition (plan §18)
----------------------
::

    config_hash = SHA256(canonical_json(config))

where ``canonical_json`` is a deterministic serialisation:

* sorted keys at every nesting level
* no insignificant whitespace
* ``ensure_ascii=False`` for human-readable UTF-8 (but the bytes are
  still stable across locales because keys are sorted)
* ``separators=(",", ":")`` so the byte stream is unique per content
* ``sort_keys=True`` via :func:`json.dumps`

The function is **pure** and side-effect-free.  Callers compose the
mapping they want to hash — this module never reaches into ``MLConfig``
directly, so it can be unit-tested with plain dicts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Final, Mapping

log = logging.getLogger(__name__)


# ── Canonical serialisation ─────────────────────────────────────────────────


def _to_plain(value: Any) -> Any:
    """Coerce a value into a JSON-serialisable, hash-stable form.

    * ``set``/``frozenset`` → ``list`` (sorted if items are sortable,
      otherwise insertion order — the latter is fine for our usage).
    * ``dataclass`` → ``dict`` via :func:`dataclasses.asdict`.
    * ``Mapping`` → ``dict`` (recursive).
    * ``tuple`` → ``list``.
    * Everything else is returned as-is.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(_to_plain(v) for v in value)
        except TypeError:
            return [_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def canonical_json(config: Mapping[str, Any] | Any) -> str:
    """Serialise ``config`` to a deterministic JSON string.

    Two equivalent inputs always produce the same byte string, regardless
    of dict insertion order, set ordering, or dataclass presence.
    """
    if isinstance(config, Mapping):
        plain = {str(k): _to_plain(v) for k, v in config.items()}
    else:
        plain = _to_plain(config)
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


# ── Hashing ─────────────────────────────────────────────────────────────────


HASH_ALGORITHM: Final[str] = "sha256"
HASH_PREFIX: Final[str] = "sha256:"


def compute_config_hash(config: Mapping[str, Any] | Any) -> str:
    """Return the canonical SHA-256 fingerprint of ``config``.

    Output format: ``"sha256:<64-hex-chars>"``.  The prefix allows
    future migration to a stronger algorithm without breaking log
    parsers.
    """
    payload = canonical_json(config)
    digest = hashlib.new(HASH_ALGORITHM, payload.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def verify_config_hash(
    config: Mapping[str, Any] | Any,
    expected: str,
) -> bool:
    """Return True iff ``compute_config_hash(config) == expected``.

    This is the gate primitive used by :mod:`ml.scripts.check_promotion_gate`
    to deny promotion when the candidate's hash differs from the report's
    recorded hash (plan §18).
    """
    if not isinstance(expected, str) or not expected:
        return False
    actual = compute_config_hash(config)
    if actual == expected:
        return True
    log.warning(
        "config_hash mismatch: expected=%s actual=%s", expected, actual
    )
    return False


def short_hash(config: Mapping[str, Any] | Any, *, length: int = 12) -> str:
    """Return a human-friendly truncated hash for logs (no prefix)."""
    full = compute_config_hash(config)
    return full[len(HASH_PREFIX) : len(HASH_PREFIX) + length]


# ── Bundle helpers ──────────────────────────────────────────────────────────


def build_config_bundle(
    *,
    config: Mapping[str, Any] | Any,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-describing bundle: payload + canonical hash.

    Use this when persisting an artefact (promotion report, shadow
    artifact, audit entry) so the hash travels with the content it
    authenticates.
    """
    bundle: dict[str, Any] = {"config": _to_plain(config)}
    if extra:
        bundle["extra"] = _to_plain(extra)
    bundle["config_hash"] = compute_config_hash(bundle["config"])
    return bundle


__all__ = [
    "HASH_ALGORITHM",
    "HASH_PREFIX",
    "canonical_json",
    "compute_config_hash",
    "verify_config_hash",
    "short_hash",
    "build_config_bundle",
]
