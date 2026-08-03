"""MANTRA role definitions: depth hierarchy, pool fusion, and helpers.

MANTRA defines 12 roles with a tactical depth hierarchy used to determine
a player's *primary* role (the most defensive slot they can cover):

    Por (0) → Dc, B, Dd, Ds (1) → E, M (2) → C (3) → T, W (4) → A, Pc (5)

Lower depth value = more defensive = higher priority when determining the
primary role for a multi-role player.
"""

from __future__ import annotations

from typing import Optional

# ── Role catalogue ───────────────────────────────────────────────────────────

ALL_ROLES: list[str] = [
    "Por", "Dc", "B", "Dd", "Ds", "E", "M", "C", "T", "W", "A", "Pc",
]

#: Tactical depth: lower = more defensive.
DEPTH_ORDER: dict[str, int] = {
    "Por": 0,
    "Dc": 1, "B": 1, "Dd": 1, "Ds": 1,
    "E": 2, "M": 2,
    "C": 3,
    "T": 4, "W": 4,
    "A": 5, "Pc": 5,
}

#: Pool fusion table: when a role has fewer than ``SOGLIA_POOL`` players,
#: the pool is widened by merging with the roles listed here.
POOL_FUSIONE: dict[str, tuple[str, ...]] = {
    "B":  ("Dc", "Dd", "Ds"),
    "Dd": ("Dc", "Ds", "B"),
    "Ds": ("Dc", "Dd", "B"),
    "E":  ("M",),
    "M":  ("E",),
    "Pc": ("A",),
    "A":  ("Pc",),
    "T":  ("W",),
    "W":  ("T",),
}

# Roles that do NOT appear in POOL_FUSIONE (Por, Dc, C) are already large
# enough or are the target of fusion from smaller roles.

# ── MANTRA role code mapping from listone rm column ─────────────────────────

#: Normalisation map for rm codes found in the Fantacalcio listone XLSX.
#: Keys are the raw values that may appear; values are the canonical
#: MANTRA role name.
MANTRA_ROLE_MAP: dict[str, str] = {
    "Por": "Por",
    "POR": "Por",
    "Dc": "Dc",
    "DC": "Dc",
    "Dd": "Dd",
    "DD": "Dd",
    "Ds": "Ds",
    "DS": "Ds",
    "B": "B",
    "E": "E",
    "M": "M",
    "C": "C",
    "T": "T",
    "W": "W",
    "A": "A",
    "Pc": "Pc",
    "PC": "Pc",
    "P": "Pc",       # rare: P used for Punta in some old listoni
}


# ── Public helpers ──────────────────────────────────────────────────────────


def calcola_ruolo_primario(ruoli: list[str]) -> Optional[str]:
    """Return the most defensive (lowest depth) role from a list.

    When two roles have equal depth, the one appearing first in
    ``ruoli`` wins (original list order).

    Parameters
    ----------
    ruoli:
        List of canonical MANTRA role codes (e.g. ``["Dd", "E"]``).

    Returns
    -------
    The primary role, or ``None`` if the list is empty or no role is
    recognised.
    """
    if not ruoli:
        return None

    best: Optional[str] = None
    best_depth: int = 999
    for r in ruoli:
        depth = DEPTH_ORDER.get(r, 999)
        if depth < best_depth:
            best_depth = depth
            best = r
        # Equal depth: keep the first encountered (preserve original order).
    return best


def calcola_pool_esteso(
    ruolo: str,
    role_counts: dict[str, int] | None = None,
    soglia: int = 20,
) -> set[str]:
    """Return the set of roles that form the statistical pool for *ruolo*.

    When *role_counts* is given and *ruolo* already has at least *soglia*
    players on its own, no fusion happens — the pool is just ``{ruolo}``,
    regardless of whether *ruolo* appears in ``POOL_FUSIONE`` (as a key or
    as someone else's fusion target). Fusion is a small-sample safeguard,
    not an unconditional merge.

    Otherwise (role_counts not supplied, or *ruolo*'s own count is below
    *soglia*): if *ruolo* has a fusion entry in ``POOL_FUSIONE``, the pool
    includes both the role itself and all fused roles. If *ruolo* is the
    target of someone else's fusion (e.g. ``"Dc"`` for ``"B"``), the pool
    includes that base role too. Otherwise the pool is just the role
    itself (e.g. ``"Por"`` → ``{"Por"}``).

    Parameters
    ----------
    ruolo:
        Canonical MANTRA role code.
    role_counts:
        Optional mapping of role code → number of players currently in
        that role. When provided, gates fusion on *ruolo*'s own sample
        size instead of merging unconditionally.
    soglia:
        Minimum sample size for *ruolo* to stand on its own (default 20,
        matching ``MantraConfig.SOGLIA_POOL``).

    Returns
    -------
    A set of role codes that belong to the same statistical pool.
    """
    if role_counts is not None and role_counts.get(ruolo, 0) >= soglia:
        return {ruolo}
    fused = POOL_FUSIONE.get(ruolo)
    if fused:
        return {ruolo} | set(fused)
    # Also check if *ruolo* is the target of someone else's fusion
    for base, targets in POOL_FUSIONE.items():
        if ruolo in targets:
            return {ruolo, base} | set(t for t in targets if t != ruolo)
    return {ruolo}


def normalizza_rm(raw: str) -> list[str]:
    """Parse a raw ``rm`` string from the listone XLSX into canonical codes.

    Examples
    --------
    >>> normalizza_rm("Dd;E")
    ['Dd', 'E']
    >>> normalizza_rm("A")
    ['A']
    >>> normalizza_rm("T;W")
    ['T', 'W']
    >>> normalizza_rm("DC;E")
    ['Dc', 'E']
    """
    if not raw or not isinstance(raw, str):
        return []
    parts = raw.replace(",", ";").replace("/", ";").split(";")
    results: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        canonical = MANTRA_ROLE_MAP.get(p)
        if canonical:
            results.append(canonical)
    return results
