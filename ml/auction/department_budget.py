"""Pre-auction department spending ceilings (hard + recommended).

Combines:
  * the feasibility rule "≥ min_slot_price credits for every still-empty slot"
    already enforced live in orchestrator.py (here generalized per department
    and projected to auction start);
  * the market prior per department derived from the listino
    (LISTINO_BUDGET_SHARE_PRIOR for CLASSIC; MANTRA_LISTINO_BUDGET_SHARE_PRIOR
    calibrated offline via scripts/calibrate_mantra_budget_share_prior.py);
  * the structural slot weight (role_quotas).

Does not introduce new infrastructure: reuses AuctionConfig, RulesetType and
the budget_initial / reference_budget scaling factor already present in
price_drift.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

from ml.auction.models import AuctionConfig
from ml.optimizer.models import RulesetType

CLASSIC_DEPARTMENTS: Final[dict[str, tuple[str, ...]]] = {
    "P": ("P",),
    "D": ("D",),
    "C": ("C",),
    "A": ("A",),
}

MANTRA_DEPARTMENTS: Final[dict[str, tuple[str, ...]]] = {
    "POR": ("Por",),
    "DIF": ("Dc", "B", "Dd", "Ds"),
    "CEN": ("E", "M", "C"),
    "TRQ": ("T", "W"),
    "ATT": ("A", "Pc"),
}

DEPARTMENT_LABELS_IT: Final[dict[str, str]] = {
    "P": "Portieri",
    "D": "Difensori",
    "C": "Centrocampisti",
    "A": "Attaccanti",
    "POR": "Portieri",
    "DIF": "Difensori",
    "CEN": "Centrocampisti",
    "TRQ": "Trequartisti/Ali",
    "ATT": "Attaccanti",
}

LISTINO_BUDGET_SHARE_PRIOR: Final[dict[str, float]] = {
    "P": 0.09,
    "D": 0.24,
    "C": 0.35,
    "A": 0.32,
}

# Calibrated from Quotazioni_Fantacalcio_Stagione_2026_27.xlsx
# via scripts/calibrate_mantra_budget_share_prior.py
MANTRA_LISTINO_BUDGET_SHARE_PRIOR: Final[dict[str, float]] = {
    "POR": 0.0975,
    "DIF": 0.2199,
    "CEN": 0.3404,
    "TRQ": 0.1013,
    "ATT": 0.2409,
}


@dataclass(frozen=True)
class DepartmentCapConfig:
    min_slot_price: int = 1
    alpha_market_vs_slot: float = 0.65
    tolerance: float = 0.20

    def __post_init__(self) -> None:
        if self.min_slot_price < 1:
            raise ValueError(f"min_slot_price must be >= 1, got {self.min_slot_price}")
        if not 0.0 <= self.alpha_market_vs_slot <= 1.0:
            raise ValueError(
                f"alpha_market_vs_slot must be in [0,1], got {self.alpha_market_vs_slot}"
            )
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError(f"tolerance must be in [0,1), got {self.tolerance}")


@dataclass(frozen=True)
class CreditsAndPercent:
    credits: int
    percent: float


@dataclass(frozen=True)
class DepartmentCap:
    department_id: str
    label_it: str
    roles: tuple[str, ...]
    slots: int
    hard_cap: CreditsAndPercent
    recommended_min: CreditsAndPercent
    recommended_max: CreditsAndPercent
    clamped_to_hard_cap: bool
    market_share_prior: float | None
    slot_share: float
    market_share_source: str


@dataclass(frozen=True)
class DepartmentBudgetPlan:
    ruleset: str
    budget_initial: int
    reference_budget: int
    total_slots: int
    min_slot_price: int
    departments: tuple[DepartmentCap, ...]
    sum_recommended_max_percent: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _credits_and_percent(credits: int, budget: int) -> CreditsAndPercent:
    if budget <= 0:
        return CreditsAndPercent(credits=credits, percent=0.0)
    return CreditsAndPercent(credits=credits, percent=round(credits / budget * 100.0, 1))


def _departments_for_ruleset(ruleset: RulesetType) -> dict[str, tuple[str, ...]]:
    if ruleset == "CLASSIC":
        return CLASSIC_DEPARTMENTS
    if ruleset == "MANTRA":
        return MANTRA_DEPARTMENTS
    raise ValueError(f"Unsupported ruleset: {ruleset!r}")


def _market_prior_for(ruleset: RulesetType, dept_id: str) -> tuple[float | None, str]:
    if ruleset == "CLASSIC" and dept_id in LISTINO_BUDGET_SHARE_PRIOR:
        return LISTINO_BUDGET_SHARE_PRIOR[dept_id], "listino_prior"
    if ruleset == "MANTRA" and dept_id in MANTRA_LISTINO_BUDGET_SHARE_PRIOR:
        return MANTRA_LISTINO_BUDGET_SHARE_PRIOR[dept_id], "listino_prior"
    return None, "fallback_slot_only"


def compute_department_budget_plan(
    config: AuctionConfig,
    cap_config: DepartmentCapConfig | None = None,
    *,
    budget_override: int | None = None,
    filled_slots_by_role: Mapping[str, int] | None = None,
) -> DepartmentBudgetPlan:
    if cap_config is None:
        cap_config = DepartmentCapConfig()

    budget = budget_override if budget_override is not None else config.budget_initial
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")

    filled = filled_slots_by_role or {}
    remaining_by_role = {
        role: max(0, quota - filled.get(role, 0))
        for role, quota in config.role_quotas.items()
    }
    total_slots = sum(remaining_by_role.values())
    if total_slots <= 0:
        raise ValueError("No remaining slots; cannot compute department budget plan")

    departments_map = _departments_for_ruleset(config.ruleset)
    min_price = cap_config.min_slot_price
    alpha = cap_config.alpha_market_vs_slot
    tol = cap_config.tolerance

    caps: list[DepartmentCap] = []
    sum_rec_max_pct = 0.0
    warnings: list[str] = []

    for dept_id, roles in departments_map.items():
        active_roles = tuple(r for r in roles if r in remaining_by_role)
        n_dept = sum(remaining_by_role.get(r, 0) for r in active_roles)
        if n_dept == 0 and not active_roles:
            continue

        slot_share = n_dept / total_slots if total_slots else 0.0
        market_share, market_source = _market_prior_for(config.ruleset, dept_id)

        if market_share is not None:
            blended = alpha * market_share + (1.0 - alpha) * slot_share
        else:
            blended = slot_share

        other_slots = total_slots - n_dept
        hard_credits = max(0, budget - other_slots * min_price)

        central = blended * budget
        rec_max_raw = round(central * (1.0 + tol))
        rec_min_raw = round(central * (1.0 - tol))

        clamped = False
        rec_max = min(rec_max_raw, hard_credits)
        rec_min = min(rec_min_raw, hard_credits)
        if rec_max < rec_max_raw or rec_min < rec_min_raw:
            clamped = True
        if rec_min > rec_max:
            rec_min = rec_max
            clamped = True

        hard_cp = _credits_and_percent(hard_credits, budget)
        rec_min_cp = _credits_and_percent(rec_min, budget)
        rec_max_cp = _credits_and_percent(rec_max, budget)
        sum_rec_max_pct += rec_max_cp.percent

        caps.append(
            DepartmentCap(
                department_id=dept_id,
                label_it=DEPARTMENT_LABELS_IT.get(dept_id, dept_id),
                roles=active_roles,
                slots=n_dept,
                hard_cap=hard_cp,
                recommended_min=rec_min_cp,
                recommended_max=rec_max_cp,
                clamped_to_hard_cap=clamped,
                market_share_prior=market_share,
                slot_share=round(slot_share, 6),
                market_share_source=market_source,
            )
        )

    if sum_rec_max_pct > 100.0 + 1e-6:
        warnings.append(
            f"sum_recommended_max_percent={sum_rec_max_pct:.1f} exceeds 100; "
            "individual department high bands are not simultaneously spendable"
        )

    return DepartmentBudgetPlan(
        ruleset=config.ruleset,
        budget_initial=budget,
        reference_budget=config.reference_budget,
        total_slots=total_slots,
        min_slot_price=min_price,
        departments=tuple(caps),
        sum_recommended_max_percent=round(sum_rec_max_pct, 1),
        warnings=tuple(warnings),
    )
