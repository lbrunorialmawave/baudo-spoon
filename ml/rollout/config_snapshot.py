"""Canonical MLConfig snapshot for cross-artifact hashing (WS16, plan §18).

Goal
----
Three persisted artefacts must carry a ``config_hash`` that is **stable
across re-emissions** of the same effective configuration:

* ``artifacts/effective_config.json`` (run_pipeline → train job)
* ``artifacts/promotion_report.json`` (run_pipeline → train job)
* ``artifacts/canary_report.json`` (run_pipeline → train job)

If the snapshot shape diverges between the three emission sites, a
re-run of the pipeline produces a different hash for the same logical
configuration and the promotion gate (Phase 6 of ``ml-training.yml``)
denies the transition with ``config_hash mismatch: report=... candidate=...``
— even though nothing actually changed.

The single source of truth lives in this module:

* :data:`ML_CONFIG_SNAPSHOT_KEYS` — the ordered tuple of field names
  that participate in the canonical hash.
* :func:`build_ml_config_snapshot` — returns a ``dict[str, Any]`` whose
  keys are exactly :data:`ML_CONFIG_SNAPSHOT_KEYS` and whose values are
  coerced into hash-stable primitive types (``int``, ``bool``, ``str``).

Effective-config callers that need additional fields (``production_mode``,
``production_flags``, ``stages``, ``test_seasons``, …) must place them
in the ``extra`` block of :func:`ml.rollout.config_hash.build_config_bundle`
so they do not perturb the canonical hash.

Idempotency note
----------------
Re-running the pipeline against the same effective configuration MUST
produce the same ``config_hash`` on every emission.  This module is
pure — no I/O, no logging side effects — and is unit-tested by
``ml/tests/test_config_snapshot.py``.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

# Ordine canonico dei campi ML che entrano nel config_hash condiviso.
# Tenere come tupla ``Final``: aggiungere un campo richiede un bump di
# versione del promotion report (e di tutti i test che confrontano
# snapshot) per evitare mismatch silenziosi.
ML_CONFIG_SNAPSHOT_KEYS: Final[tuple[str, ...]] = (
    "min_minutes",
    "min_minutes_hard",
    "enable_limited_sample_training",
    "enable_shrinkage",
    "enable_recent_role_features",
    "enable_breakout_model",
    "weighting_strategy",
    "shrinkage_prior_strength",
    "reliability_weight_mode",
)


def _coerce(value: Any) -> Any:
    """Coerce a single :class:`MLConfig` field into a hash-stable primitive.

    Centralised so the canary, the promotion report, and the effective
    config all use the same canonical types (``int`` vs ``bool`` vs ``str``).
    """
    # Bool check MUST come first because ``bool`` is a subclass of ``int``.
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def build_ml_config_snapshot(cfg: Any) -> dict[str, Any]:
    """Return the canonical MLConfig snapshot for hashing.

    Args:
        cfg: An :class:`ml.config.MLConfig` instance, or any object that
            exposes the fields listed in :data:`ML_CONFIG_SNAPSHOT_KEYS`.
            The function reads attributes only — it never imports the
            concrete class — so unit tests can pass lightweight stubs.

    Returns:
        A ``dict`` keyed by :data:`ML_CONFIG_SNAPSHOT_KEYS`, in the
        same order.  Every value is coerced to a hash-stable primitive.

    Raises:
        AttributeError: if a required field is missing on ``cfg``.
    """
    snapshot: dict[str, Any] = {}
    for key in ML_CONFIG_SNAPSHOT_KEYS:
        if not hasattr(cfg, key):
            raise AttributeError(
                f"MLConfig snapshot is missing required field {key!r}. "
                "Update ML_CONFIG_SNAPSHOT_KEYS and bump the report "
                "version if the schema changed intentionally."
            )
        snapshot[key] = _coerce(getattr(cfg, key))
    return snapshot


def merge_ml_snapshot(
    cfg: Any,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: snapshot + caller-provided extra fields.

    The returned dict preserves :data:`ML_CONFIG_SNAPSHOT_KEYS` order
    and appends ``extra`` keys in insertion order at the end.  This
    matches how :func:`ml.rollout.config_hash.build_config_bundle`
    expects its ``config`` argument.
    """
    merged: dict[str, Any] = build_ml_config_snapshot(cfg)
    if extra:
        for k, v in extra.items():
            if k in merged:
                # Extra MUST NOT shadow the canonical snapshot — that
                # would re-introduce the drift this module exists to
                # prevent.
                raise ValueError(
                    f"extra key {k!r} collides with the canonical ML "
                    "snapshot; rename it or update ML_CONFIG_SNAPSHOT_KEYS."
                )
            merged[k] = v
    return merged


__all__ = [
    "ML_CONFIG_SNAPSHOT_KEYS",
    "build_ml_config_snapshot",
    "merge_ml_snapshot",
]
