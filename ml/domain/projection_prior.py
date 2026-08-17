"""Role-based projection priors for listino players without ML coverage.

Context
-------
``DataRepository.get_player_pool`` joins Fantacalcio quotations to the ML
prediction artifact.  Players present on the listino but absent from the
artifact (or outside the plausible voto range) were previously *dropped*
from the auction/optimizer pool (``excluded_no_projection``).

For a live auction that must cover the full listino, that hard exclusion
blocks the giro.  This module supplies an explicit, auditable fallback:

* **Never** use Fantacalcio FVM as a voto proxy (different scale).
* Use a **role prior** in the Fantacalcio single-match rating band [3, 10].
* Tag provenance (``projection_source="role_prior"``) and damp decisions
  via ``INSUFFICIENT`` cohort + floor reliability weight.

Policy is opt-in at the call site (auction vs optimizer) so the ILP does
not silently optimise on priors unless the caller chooses so.
"""

from __future__ import annotations

from typing import Final, Literal, Mapping

# Optimizer / auction role codes (Fantacalcio classic).
RoleCode = Literal["P", "D", "C", "A"]

UnprojectedPolicy = Literal["exclude", "role_prior"]

# Conservative per-match voto priors.  Anchored near the classic "sufficient"
# band so unprojected players are callable in auction without looking like
# top-tier projections.  Tunable via ROLE_PRIOR_SCORE overrides in tests.
DEFAULT_ROLE_PRIOR_SCORE: Final[Mapping[RoleCode, float]] = {
    "P": 6.0,
    "D": 6.0,
    "C": 6.1,
    "A": 6.2,
}

# Plausible single-match Fantacalcio voto band (same as get_player_pool).
MIN_PLAUSIBLE_SCORE: Final[float] = 3.0
MAX_PLAUSIBLE_SCORE: Final[float] = 10.0

PROJECTION_SOURCE_ML: Final[str] = "ml"
PROJECTION_SOURCE_ROLE_PRIOR: Final[str] = "role_prior"


def role_prior_score(
    role: str,
    *,
    priors: Mapping[str, float] | None = None,
) -> float:
    """Return a prior projected_score for ``role``.

    Raises:
        ValueError: if ``role`` is not a known optimizer role code.
    """
    table = priors or DEFAULT_ROLE_PRIOR_SCORE
    if role not in table:
        raise ValueError(
            f"role_prior_score: unknown role {role!r}; "
            f"expected one of {sorted(table)}"
        )
    score = float(table[role])
    if not (MIN_PLAUSIBLE_SCORE <= score <= MAX_PLAUSIBLE_SCORE):
        raise ValueError(
            f"role_prior_score: prior {score} for role {role!r} outside "
            f"[{MIN_PLAUSIBLE_SCORE}, {MAX_PLAUSIBLE_SCORE}]"
        )
    return score


def is_plausible_projection(score: float | None) -> bool:
    """True iff ``score`` is a usable in-band Fantacalcio voto proxy."""
    if score is None:
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    if value <= 0.0:
        return False
    return MIN_PLAUSIBLE_SCORE <= value <= MAX_PLAUSIBLE_SCORE
