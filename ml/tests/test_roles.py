from __future__ import annotations

import logging

import pytest

from ml.data.roles import CanonicalRole, fotmob_key_to_role, get_player_role

_ROLES_LOGGER = "ml.data.roles"


# ── fotmob_key_to_role ────────────────────────────────────────────────────────

def test_goalkeeper_maps_to_gk() -> None:
    assert fotmob_key_to_role("goalkeeper") == "GK"


@pytest.mark.parametrize("key", [
    "right-back", "left-back", "centre-back",
    "right-centre-back", "left-centre-back",
    "right-wing-back", "left-wing-back",
    "rightback", "leftback", "centreback",
])
def test_defender_variants_map_to_def(key: str) -> None:
    assert fotmob_key_to_role(key) == "DEF"


@pytest.mark.parametrize("key", [
    "defensive-midfielder", "central-midfielder",
    "right-midfielder", "left-midfielder", "attacking-midfielder",
    "defensivemidfielder", "centralmidfielder",
])
def test_midfielder_variants_map_to_mid(key: str) -> None:
    assert fotmob_key_to_role(key) == "MID"


@pytest.mark.parametrize("key", [
    "striker", "second-striker", "winger",
    "right-winger", "left-winger", "rightwinger", "leftwinger",
])
def test_forward_variants_map_to_fwd(key: str) -> None:
    assert fotmob_key_to_role(key) == "FWD"


def test_normalisation_strips_hyphens(caplog: pytest.LogCaptureFixture) -> None:
    assert fotmob_key_to_role("Right-Back") == "DEF"
    assert fotmob_key_to_role("right-back") == "DEF"
    assert fotmob_key_to_role("rightback") == "DEF"


def test_unknown_key_falls_back_to_fwd_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_ROLES_LOGGER):
        result = fotmob_key_to_role("beachvolleyplayer")
    assert result == "FWD"
    assert any("unrecognized" in r.message for r in caplog.records)


def test_none_key_falls_back_to_fwd_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_ROLES_LOGGER):
        result = fotmob_key_to_role(None)
    assert result == "FWD"
    assert caplog.records


# ── get_player_role ───────────────────────────────────────────────────────────

def test_get_player_role_primary_path() -> None:
    desc = {"primaryPosition": {"key": "goalkeeper"}, "nonPrimaryPositions": []}
    assert get_player_role(desc) == "GK"


def test_get_player_role_nonprimary_fallback() -> None:
    desc = {
        "primaryPosition": {"key": "unknownposition"},
        "nonPrimaryPositions": [{"key": "striker"}],
    }
    assert get_player_role(desc) == "FWD"


def test_get_player_role_positions_array_fallback() -> None:
    desc = {
        "primaryPosition": None,
        "nonPrimaryPositions": [],
        "positions": [
            {"isMainPosition": False, "strPos": {"key": "rightwinger"}},
            {"isMainPosition": True, "strPos": {"key": "striker"}},
        ],
    }
    assert get_player_role(desc) == "FWD"


def test_get_player_role_none_input_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_ROLES_LOGGER):
        result = get_player_role(None)
    assert result == "FWD"
    assert caplog.records


def test_get_player_role_real_response_structure() -> None:
    """Test with the actual structure from response/response-player.json."""
    desc = {
        "positions": [
            {"strPos": {"key": "rightwinger"}, "isMainPosition": False},
            {"strPos": {"key": "striker"}, "isMainPosition": True},
        ],
        "primaryPosition": {"label": "Striker", "key": "striker"},
        "nonPrimaryPositions": [{"label": "Right Winger", "key": "rightwinger"}],
    }
    assert get_player_role(desc) == "FWD"
