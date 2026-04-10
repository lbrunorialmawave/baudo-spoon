from __future__ import annotations

"""Bridge between FotMob playerData API response and the player_profiles schema.

Kept in the scraper package so the scraper container has no dependency on the
ml package. The role-mapping logic is intentionally duplicated (small, stable)
rather than importing from ml.data.roles.
"""

from typing import Any

_FOTMOB_TO_ROLE: dict[str, str] = {
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

def _normalize(key: str) -> str:
    return key.replace("-", "").replace(" ", "").lower()


def _resolve_role(key: str | None) -> str | None:
    if not key:
        return None
    return _FOTMOB_TO_ROLE.get(_normalize(key))


def _find_role_in_list(entries: list[Any], key_path: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Walk a list of dicts, extract a key via key_path, return (raw_key, role) or (None, None)."""
    for entry in entries or []:
        obj: Any = entry
        for part in key_path:
            obj = (obj or {}).get(part)
        role = _resolve_role(obj)
        if role:
            return obj, role
    return None, None


def extract_profile_from_player_data(
    player_id: int,
    player_name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Extract a player_profiles-compatible dict from a playerData API response.

    Tries primaryPosition → nonPrimaryPositions → positions[isMain], falls back
    to canonical_role=None (loader.py fills it as 'FWD').
    """
    position_desc: dict[str, Any] = data.get("positionDescription") or {}

    # 1. primaryPosition
    primary_key = (position_desc.get("primaryPosition") or {}).get("key")
    role_key, canonical_role = primary_key, _resolve_role(primary_key)

    # 2. nonPrimaryPositions
    if not canonical_role:
        role_key, canonical_role = _find_role_in_list(
            position_desc.get("nonPrimaryPositions") or [],
            ("key",),
        )

    # 3. positions[isMainPosition]
    if not canonical_role:
        main_positions = [
            p for p in (position_desc.get("positions") or [])
            if (p or {}).get("isMainPosition")
        ]
        role_key, canonical_role = _find_role_in_list(main_positions, ("strPos", "key"))

    return {
        "player_fotmob_id": player_id,
        "player_name": player_name,
        "role_key": role_key,
        "canonical_role": canonical_role,
    }
