"""Merge MANTRA results with ML pipeline predictions.

The merger aligns the two datasets on ``player_fotmob_id`` (primary key) and
falls back to normalised player-name matching when the ID is unavailable.

Edge cases handled
------------------
*   ``results_latest.json`` does not exist → all players marked without ML data
*   A player exists in MANTRA but not in ML → ``has_ml_data=false``, no crash
*   Multiple ML records for the same ``player_fotmob_id`` → first match wins
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _normalise(name: str) -> str:
    """Lower-case, strip, and collapse whitespace for fuzzy matching."""
    import re
    return re.sub(r"\s+", " ", name.strip().lower())


def merge_datasets(mantra_path: Path, ml_path: Path) -> dict[str, Any]:
    """Merge MANTRA and ML JSON artefacts into a single enriched structure.

    Parameters
    ----------
    mantra_path:
        Path to ``mantra_results_{season}.json``.
    ml_path:
        Path to ``results_latest.json``.

    Returns
    -------
    dict with keys:
        - ``players``: MANTRA player list enriched with ML fields where matched
        - ``meta``: merged metadata from both artefacts
        - ``mantra_classifications``: original MANTRA Fase 7/8 classifications
    """
    # ── Load MANTRA ───────────────────────────────────────────────────────────
    with mantra_path.open("r", encoding="utf-8") as f:
        mantra: dict[str, Any] = json.load(f)

    mantra_players: list[dict[str, Any]] = mantra.get("players", [])
    mantra_meta: dict[str, Any] = mantra.get("meta", {})
    mantra_classifications: dict[str, Any] = mantra.get("classifications", {})

    # ── Load ML (optional — no crash if missing) ──────────────────────────────
    ml_predictions: list[dict[str, Any]] = []
    ml_var_results: list[dict[str, Any]] = []
    ml_next_season: list[dict[str, Any]] = []
    ml_run_id: str | None = None
    ml_meta: dict[str, Any] = {}

    if ml_path.exists():
        with ml_path.open("r", encoding="utf-8") as f:
            ml_raw: dict[str, Any] = json.load(f)

        ml_predictions = ml_raw.get("predictions", [])
        ml_var_results = ml_raw.get("var_results", [])
        ml_next_season = ml_raw.get("next_season_predictions", [])
        ml_run_id = ml_raw.get("run_id")
        ml_meta = ml_raw.get("metadata", {})
        log.info("ML artefact loaded: %d predictions, %d VAR, %d next-season",
                 len(ml_predictions), len(ml_var_results), len(ml_next_season))
    else:
        log.warning("ML artefact not found at %s — all players will have has_ml_data=false", ml_path)

    # ── Index ML data by player_fotmob_id ─────────────────────────────────────
    # Also build a name-based index for fallback matching.
    ml_by_id: dict[int, dict[str, Any]] = {}
    ml_by_name: dict[str, dict[str, Any]] = {}

    for pred in ml_predictions:
        pid = pred.get("player_fotmob_id")
        if pid is not None:
            if pid not in ml_by_id:
                ml_by_id[int(pid)] = pred

        pname = pred.get("player_name")
        if pname:
            key = _normalise(str(pname))
            if key not in ml_by_name:
                ml_by_name[key] = pred

    # Index VAR by player_id (stored as "fm-{fotmob_id}" in VAR output)
    var_by_id: dict[int, dict[str, Any]] = {}
    for var in ml_var_results:
        pid_str = var.get("player_id", "")
        if pid_str and pid_str.startswith("fm-"):
            try:
                fotmob = int(pid_str[3:])
                var_by_id[fotmob] = var
            except ValueError:
                pass

    # Index next-season by name
    ns_by_name: dict[str, dict[str, Any]] = {}
    for ns in ml_next_season:
        pname = ns.get("playerName") or ns.get("player_name")
        if pname:
            ns_by_name[_normalise(str(pname))] = ns

    # ── Enrich each MANTRA player ─────────────────────────────────────────────
    match_by_id = 0
    match_by_name = 0
    no_ml = 0

    for player in mantra_players:
        pid = player.get("player_fotmob_id")
        pname = _normalise(str(player.get("player_name", "")))
        pteam = _normalise(str(player.get("team", "")))

        matched: dict[str, Any] | None = None
        match_key: str | None = None

        # 1. Primary: match by fotmob_id
        if pid is not None:
            matched = ml_by_id.get(int(pid))
            if matched is not None:
                match_by_id += 1
                match_key = "fotmob_id"

        # 2. Fallback: match by normalised name + team
        if matched is None:
            candidate = ml_by_name.get(pname)
            if candidate is not None:
                cand_team = _normalise(str(candidate.get("teamName", candidate.get("team_name", ""))))
                if not pteam or not cand_team or pteam == cand_team:
                    matched = candidate
                    match_by_name += 1
                    match_key = "name"

        if matched is not None:
            player["has_ml_data"] = True
            player["predicted_fantavoto"] = matched.get("predicted") or matched.get("predicted_fantavoto")
            player["prediction_std"] = matched.get("prediction_std")
            player["expected_minutes"] = matched.get("expected_minutes") or matched.get("expectedMinutes")

            # Enrich with VAR data
            if pid is not None and int(pid) in var_by_id:
                v = var_by_id[int(pid)]
                player["var_score"] = v.get("var_score")
                player["esv"] = v.get("esv")

            # Enrich with next-season prediction
            ns = ns_by_name.get(pname)
            if ns is not None:
                player["next_season_predicted"] = ns.get("predictedNextFantavoto") or ns.get("predicted_next_fantavoto")
        else:
            player["has_ml_data"] = False
            player["predicted_fantavoto"] = None
            player["prediction_std"] = None
            player["expected_minutes"] = None
            player["var_score"] = None
            player["esv"] = None
            player["next_season_predicted"] = None
            no_ml += 1

    log.info(
        "Merge complete: %d matched by fotmob_id, %d by name, %d without ML data",
        match_by_id, match_by_name, no_ml,
    )

    # ── Assemble merged meta ──────────────────────────────────────────────────
    merged_meta: dict[str, Any] = {
        "season_start": mantra_meta.get("season_start"),
        "generated_at": mantra_meta.get("generated_at"),
        "n_players_mantra": len(mantra_players),
        "n_players_with_ml": match_by_id + match_by_name,
        "n_players_without_ml": no_ml,
        "mantra_run_id": mantra_meta.get("run_id"),
        "ml_run_id": ml_run_id,
        "mantra_config": mantra_meta.get("config"),
    }

    return {
        "players": mantra_players,
        "meta": merged_meta,
        "mantra_classifications": mantra_classifications,
    }
