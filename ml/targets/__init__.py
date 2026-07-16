"""Target construction pipeline for the player evaluation system.

Exports:
- TargetBuilder: builds all 6 target columns from a player-season DataFrame
- TheoreticalFantavoto: role-aware theoretical score as a feature
- TARGET_SPECS: all 6 TargetSpec instances
"""
from ml.targets.builder import TargetBuilder, TARGET_SPECS
from ml.targets.theoretical import TheoreticalFantavoto

__all__ = ["TargetBuilder", "TARGET_SPECS", "TheoreticalFantavoto"]
