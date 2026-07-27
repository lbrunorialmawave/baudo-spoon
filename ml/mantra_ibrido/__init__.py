"""MANTRA+ML Hybrid scoring — merges MANTRA pillars with ML predictions.

Public API
---------
run_hybrid_computation  — Orchestrate merge → scoring → classifications → output
load_config            — Read persisted hybrid config from disk
update_config          — Partial-merge update of persisted config (never resets custom fields)
MantraIbridoConfig     — Config dataclass (frozen, validated at construction)
"""

from __future__ import annotations

from .config import MantraIbridoConfig
from .config_store import load_config, update_config
from .runner import run_hybrid_computation

__all__ = [
    "MantraIbridoConfig",
    "load_config",
    "update_config",
    "run_hybrid_computation",
]
