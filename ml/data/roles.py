from __future__ import annotations

"""Player role taxonomy and FotMob position key mapping.

Canonical roles:
    GK  — Goalkeeper
    DEF — Defender (centre-back, full-back, wing-back)
    MID — Midfielder (defensive, central, attacking)
    FWD — Forward (striker, winger, second striker)

Usage::

    from ml.data.roles import fotmob_key_to_role, get_player_role

    role = fotmob_key_to_role("centre-back")   # → "DEF"
    role = get_player_role(player_data["positionDescription"])  # → "DEF"
"""

import logging
from typing import Any, Literal

log = logging.getLogger(__name__)

CanonicalRole = Literal["GK", "DEF", "MID", "FWD"]

# Known FotMob positionDescription.primaryPosition.key values.
# Keys are normalised (hyphens stripped, lowercase) before lookup.
_FOTMOB_TO_ROLE: dict[str, CanonicalRole] = {
    # ── Goalkeeper ────────────────────────────────────────────────────────────
    "keeper_long": "GK",
    # ── Defenders ─────────────────────────────────────────────────────────────
    "rightback": "DEF",
    "leftback": "DEF",
    "centerback": "DEF",
    "right_wing_back": "DEF",
    "left_wing_back": "DEF", 
    "defender": "DEF",
    # ── Midfielders ───────────────────────────────────────────────────────────
    "centerdefensivemidfielder": "MID",
    "rightmidfielder": "MID",
    "leftmidfielder": "MID",
    "midfielder": "MID",
    "centermidfielder": "MID", 
    # ── Forwards ──────────────────────────────────────────────────────────────
    "striker": "FWD",
    "centerattackingmidfielder": "FWD",
    "secondstriker": "FWD",
    "rightwinger": "FWD",
    "leftwinger": "FWD",
}

_FALLBACK_ROLE: CanonicalRole = "MID"


def _normalize_key(key: str) -> str:
    """Strip hyphens/spaces and lowercase a FotMob position key."""
    return key.replace("-", "").replace(" ", "").lower()


def fotmob_key_to_role(key: str | None) -> CanonicalRole:
    """Map a FotMob position key to a CanonicalRole.

    Normalises the key (strips hyphens, lowercases) before lookup.
    Falls back to 'FWD' with a WARNING when the key is absent or unknown.

    Args:
        key: Raw positionDescription.primaryPosition.key from FotMob API.

    Returns:
        One of 'GK', 'DEF', 'MID', 'FWD'.
    """
    if key is None:
        log.warning("role_mapping: key is None, falling back to %s", _FALLBACK_ROLE)
        return _FALLBACK_ROLE

    normalized = _normalize_key(key)
    role = _FOTMOB_TO_ROLE.get(normalized)
    if role is None:
        log.warning(
            "role_mapping: unrecognized key=%r (normalized=%r), falling back to %s",
            key,
            normalized,
            _FALLBACK_ROLE,
        )
        return _FALLBACK_ROLE
    return role


def _resolve_key_from_dict(d: Any) -> str | None:
    """Safely extract a FotMob position key string from a dict."""
    if not isinstance(d, dict):
        return None
    return d.get("key") or None


def _first_mapped_role(keys: list[str | None]) -> CanonicalRole | None:
    """Return the first CanonicalRole found for any of the given keys."""
    for key in keys:
        if not key:
            continue
        role = _FOTMOB_TO_ROLE.get(_normalize_key(key))
        if role is not None:
            return role
    return None


def get_player_role(position_desc: dict[str, Any] | None) -> CanonicalRole:
    """Extract canonical role from a FotMob positionDescription dict.

    Resolution order:
    1. positionDescription.primaryPosition.key
    2. positionDescription.nonPrimaryPositions[0].key
    3. positionDescription.positions[i].strPos.key where isMainPosition == True
    4. Fallback to 'FWD' with WARNING

    Args:
        position_desc: The ``positionDescription`` value from
            ``/api/data/playerData`` response, or None.

    Returns:
        One of 'GK', 'DEF', 'MID', 'FWD'.
    """
    if not isinstance(position_desc, dict):
        log.warning("role_mapping: positionDescription is None or invalid, falling back to %s", _FALLBACK_ROLE)
        return _FALLBACK_ROLE

    # 1. primaryPosition.key
    primary_key = _resolve_key_from_dict(position_desc.get("primaryPosition"))
    role = _first_mapped_role([primary_key])
    if role is not None:
        return role

    # 2. nonPrimaryPositions — first entry with a mapped role
    non_primary: list[Any] = position_desc.get("nonPrimaryPositions") or []
    role = _first_mapped_role([_resolve_key_from_dict(e) for e in non_primary])
    if role is not None:
        log.debug("role_mapping: resolved via nonPrimaryPositions")
        return role

    # 3. positions array — isMainPosition == True entries
    positions: list[Any] = position_desc.get("positions") or []
    main_keys = [
        _resolve_key_from_dict((p or {}).get("strPos"))
        for p in positions
        if isinstance(p, dict) and p.get("isMainPosition")
    ]
    role = _first_mapped_role(main_keys)
    if role is not None:
        log.debug("role_mapping: resolved via positions[isMain]")
        return role

    log.warning(
        "role_mapping: all position paths exhausted for keys %s, falling back to %s",
        [primary_key, *[_resolve_key_from_dict(e) for e in non_primary]],
        _FALLBACK_ROLE,
    )
    return _FALLBACK_ROLE
