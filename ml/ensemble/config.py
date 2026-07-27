"""Versioned ensemble weight configuration.

Load a named config from JSON (same pattern as other configs in the project).
Swap config files to compare ensemble versions without code changes.

# ponytail: weights normalized at runtime; upgrade to per-source weights per
# role when >3 sources are live and per-role calibration data exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["EnsembleWeightConfig", "load_config", "DEFAULT_CONFIG"]


@dataclass(frozen=True)
class EnsembleWeightConfig:
    version: str
    ml_model_weight: float = 0.5
    bookmaker_weight: float = 0.3
    expert_weight: float = 0.2

    def normalized(self) -> "EnsembleWeightConfig":
        """Return a copy with weights summing to 1.0."""
        total = self.ml_model_weight + self.bookmaker_weight + self.expert_weight
        if total <= 0:
            raise ValueError("At least one weight must be > 0")
        return EnsembleWeightConfig(
            version=self.version,
            ml_model_weight=self.ml_model_weight / total,
            bookmaker_weight=self.bookmaker_weight / total,
            expert_weight=self.expert_weight / total,
        )


DEFAULT_CONFIG = EnsembleWeightConfig(version="v1.0")


def load_config(path: str | Path) -> EnsembleWeightConfig:
    """Load an EnsembleWeightConfig from a JSON file."""
    data = json.loads(Path(path).read_text())
    return EnsembleWeightConfig(**data)
