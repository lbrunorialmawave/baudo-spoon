"""Persistent config store for MantraIbridoConfig.

Reads/writes ``config/mantra_ibrido_config.json`` with atomic-write safety.

Key design decisions
--------------------
*   ``load_config()`` returns ``DEFAULTS`` when no file exists (first-run
    graceful degradation).
*   ``update_config(partial)`` merges *partial* against the **currently
    persisted** config, **not** against ``DEFAULTS``.  This prevents an
    innocuous one-field update from silently resetting previously customised
    weights back to factory defaults.
*   Writes are atomic (write to temporary file → ``os.replace``) – a crash
    mid-write never leaves a truncated or corrupted config behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import MantraIbridoConfig

DEFAULT_CONFIG_PATH = Path("config/mantra_ibrido_config.json")

# ── Factory defaults (used only as fallback when no file exists) ──────────────

DEFAULTS: dict[str, Any] = {
    "PESO_MANTRA": 0.5,
    "PESO_ML": 0.5,
    "W_PREDICTION_STD": 0.6,
    "W_MINUTES": 0.4,
    "EV_SCALE_FACTOR": 1.0,
    "CONFIDENZA_SOGLIA": 57.0,
    "ML_BOOST_SOGLIA": 70.0,
    "ML_BOOST_FP_CORR_MAX": 60.0,
    "ML_TOP_PRED_MIN": 6.7,
    "ML_TOP_BOOST_MIN": 65.0,
    "SOGLIA_GAP_ALERT": 30.0,
    "SLEEPER_FP_CORR_MAX": 30.0,
    "SLEEPER_ML_NORM_MIN": 45.0,
    "BEST_VALUE_VR_MIN": 110.0,
    "BEST_VALUE_FP_IBRIDO_MIN": 50.0,
    "MINUTES_RISK_MAX": 900.0,
}

# ── Validation ────────────────────────────────────────────────────────────────


def validate_config(data: dict[str, Any]) -> None:
    """Raise ``ValueError`` if *data* violates any invariant."""
    # -- Sum constraints -------------------------------------------------------
    if not (0.999 <= data["PESO_MANTRA"] + data["PESO_ML"] <= 1.001):
        raise ValueError("PESO_MANTRA + PESO_ML deve essere 1.0")
    if not (0.999 <= data["W_PREDICTION_STD"] + data["W_MINUTES"] <= 1.001):
        raise ValueError("W_PREDICTION_STD + W_MINUTES deve essere 1.0")

    # -- Range constraints -----------------------------------------------------
    for k in ("PESO_MANTRA", "PESO_ML", "W_PREDICTION_STD", "W_MINUTES"):
        v = data[k]
        if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} deve essere in [0, 1], got {v!r}")

    for k in ("CONFIDENZA_SOGLIA", "ML_BOOST_SOGLIA", "ML_BOOST_FP_CORR_MAX",
              "ML_TOP_PRED_MIN", "ML_TOP_BOOST_MIN",
              "SOGLIA_GAP_ALERT", "EV_SCALE_FACTOR",
              "SLEEPER_FP_CORR_MAX", "SLEEPER_ML_NORM_MIN",
              "BEST_VALUE_VR_MIN", "BEST_VALUE_FP_IBRIDO_MIN",
              "MINUTES_RISK_MAX"):
        v = data[k]
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{k} deve essere > 0, got {v!r}")


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _atomic_write(data: dict[str, Any], path: Path) -> None:
    """Serialise *data* to *path* via atomic temporary-file write.

    A crash after the ``json.dump`` but before ``os.replace`` leaves the
    original file intact; a crash before ``json.dump`` is harmless because
    the temporary file is discarded when the file descriptor is closed.
    """
    os.makedirs(path.parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        # If something went wrong, clean up the temp file so we don't leak it.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Public API ────────────────────────────────────────────────────────────────


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> MantraIbridoConfig:
    """Read persisted config; return ``DEFAULTS`` if *path* does not exist.

    Any keys missing from the file are filled from ``DEFAULTS`` so that
    adding a new parameter in a future release does not break old configs.
    """
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        full = {**DEFAULTS, **data}
    else:
        full = dict(DEFAULTS)

    validate_config(full)
    return MantraIbridoConfig(**full)


def save_config(config: MantraIbridoConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist *config* to *path* (atomic write)."""
    data = asdict(config)
    validate_config(data)
    _atomic_write(data, path)


def update_config(partial: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> MantraIbridoConfig:
    """Merge *partial* into the **currently persisted** config and save.

    .. important::

        The merge is performed against the config on disk, **not** against
        ``DEFAULTS``.  This guarantees that a one-field update (e.g. changing
        ``EV_SCALE_FACTOR``) does not silently reset previously customised
        ``PESO_MANTRA`` / ``PESO_ML`` back to 0.5/0.5.

    Returns the new ``MantraIbridoConfig``.
    """
    current = asdict(load_config(path))
    merged = {**current, **partial}
    validate_config(merged)
    new_config = MantraIbridoConfig(**merged)
    save_config(new_config, path)
    return new_config
