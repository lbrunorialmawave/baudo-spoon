"""Target construction pipeline for the player evaluation system.

Exports:
- TargetBuilder: builds all 6 target columns from a player-season DataFrame
- TheoreticalFantavoto: role-aware theoretical score as a feature
- TARGET_SPECS: all 6 TargetSpec instances
"""

from ml.targets.builder import TARGET_SPECS, TargetBuilder
from ml.targets.theoretical import TheoreticalFantavoto

__all__ = ["TARGET_SPECS", "TargetBuilder", "TheoreticalFantavoto"]
