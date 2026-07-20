"""Schedule difficulty coefficient computation.

.. deprecated::
    Not wired into the training pipeline. See ``coefficients.py`` for details.

Public API:
    ``compute_difficulty_coefficients`` — compute per-row normalised difficulty
    coefficients from opponent strength columns.
"""
from ml.schedule_adjustment.coefficients import compute_difficulty_coefficients

__all__ = ["compute_difficulty_coefficients"]
