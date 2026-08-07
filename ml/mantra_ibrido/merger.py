"""Merge MANTRA results with ML pipeline predictions.

The merger aligns the two datasets on ``player_fotmob_id`` (primary key) and
falls back to normalised player-name matching when the ID is unavailable.

An optional **id-map file** (exported from ``player_id_map`` DB table) bridges
``fantacalcio_id → player_fotmob_id`` so that players without a pre-resolved
``player_fotmob_id`` in the MANTRA artefact can still be matched.

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
    """Lower-case, strip, drop accents/apostrophes/hyphens, collapse whitespace."""
    import re
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    # Drop apostrophes, hyphens, dots so "D'Ambrosio" → "dambrosio", "De Ketelaere" → "de ketelaere"
    ascii_str = re.sub(r"['`.\-]", "", ascii_str)
    return re.sub(r"\s+", " ", ascii_str.strip().lower())


def load_id_map(id_map_path: Path | str | None) -> dict[int, int]:
    """Load a ``{fantacalcio_id: player_fotmob_id}`` mapping from a JSON file.

    Expected JSON format (list of objects):
    ``[{"fantacalcio_id": 123, "player_fotmob_id": 456, ...}, ...]``
    """
    if id_map_path is None:
        return {}
    path = Path(id_map_path) if isinstance(id_map_path, str) else id_map_path
    if not path.exists():
        log.warning("ID map not found at %s — skipping fantacalcio_id bridge", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows: list[dict[str, Any]] = json.load(f)
    mapping: dict[int, int] = {}
    for row in rows:
        fc_id = row.get("fantacalcio_id")
        fm_id = row.get("player_fotmob_id")
        if fc_id is not None and fm_id is not None:
            mapping[int(fc_id)] = int(fm_id)
    log.info("Loaded ID map: %d entries", len(mapping))
    return mapping


def merge_datasets(
    mantra_path: Path,
    ml_path: Path,
    id_map_path: Path | None = None,
) -> dict[str, Any]:
    """Merge MANTRA and ML JSON artefacts into a single enriched structure.

    Parameters
    ----------
    mantra_path:
        Path to ``mantra_results_{season}.json``.
    ml_path:
        Path to ``results_latest.json``.
    id_map_path:
        Optional path to a JSON file with the ``player_id_map`` table contents.
        When provided, the merger uses it to resolve ``fantacalcio_id →
        player_fotmob_id`` for MANTRA players that lack the field.

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

    # ── Load optional ID map ──────────────────────────────────────────────────
    fc_to_fm: dict[int, int] = load_id_map(id_map_path)

    # ── Load ML (optional — no crash if missing) ──────────────────────────────
    ml_predictions: list[dict[str, Any]] = []
    ml_var_results: list[dict[str, Any]] = []
    ml_next_season: list[dict[str, Any]] = []
    ml_run_id: str | None = None

    if ml_path.exists():
        with ml_path.open("r", encoding="utf-8") as f:
            ml_raw: dict[str, Any] = json.load(f)

        ml_predictions = ml_raw.get("predictions", [])
        ml_var_results = ml_raw.get("var_results", [])
        ml_next_season = ml_raw.get("next_season_predictions", [])
        ml_run_id = ml_raw.get("run_id")
        log.info(
            "ML artefact loaded: %d predictions, %d VAR, %d next-season",
            len(ml_predictions),
            len(ml_var_results),
            len(ml_next_season),
        )
    else:
        log.warning(
            "ML artefact not found at %s — all players will have has_ml_data=false",
            ml_path,
        )

    # ── Index ML data by player_fotmob_id ─────────────────────────────────────
    # Also build a name-based index for fallback matching.
    ml_by_id: dict[int, dict[str, Any]] = {}
    ml_by_name: dict[str, dict[str, Any]] = {}
    ml_by_surname: dict[str, list[dict[str, Any]]] = {}

    for pred in ml_predictions:
        pid = pred.get("player_fotmob_id")
        if pid is not None and pid not in ml_by_id:
            ml_by_id[int(pid)] = pred

        pname = pred.get("player_name")
        if pname:
            key = _normalise(str(pname))
            if key not in ml_by_name:
                ml_by_name[key] = pred
            # Also index by surname (last token) for fallback matching
            tokens = key.split()
            if tokens:
                surname = tokens[-1]
                ml_by_surname.setdefault(surname, []).append(pred)

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
    match_by_bridge = 0
    no_ml = 0

    for player in mantra_players:
        pid = player.get("player_fotmob_id")
        fc_id = player.get("fantacalcio_id")

        # Bridge via ID map if direct fotmob_id is missing
        if pid is None and fc_id is not None and int(fc_id) in fc_to_fm:
            pid = fc_to_fm[int(fc_id)]
            player["player_fotmob_id"] = pid  # enrich the player dict
            log.debug(
                "Bridged fantacalcio_id=%s → fotmob_id=%s for %s",
                fc_id,
                pid,
                player.get("player_name"),
            )

        pname = _normalise(str(player.get("player_name", "")))
        pteam = _normalise(str(player.get("team", "")))

        matched: dict[str, Any] | None = None

        # 1. Primary: match by fotmob_id
        if pid is not None:
            matched = ml_by_id.get(int(pid))
            if matched is not None:
                match_by_id += 1

        # 2. Fallback: match by normalised name + team
        if matched is None:
            candidate = ml_by_name.get(pname)
            if candidate is not None:
                cand_team = _normalise(
                    str(candidate.get("teamName", candidate.get("team_name", "")))
                )
                if not pteam or not cand_team or pteam == cand_team:
                    matched = candidate
                    if pid is not None and int(pid) == candidate.get(
                        "player_fotmob_id"
                    ):
                        match_by_bridge += 1
                    else:
                        match_by_name += 1

        # 3. Fallback: match by surname only (last name token)
        if matched is None:
            surname = pname.split()[-1] if pname else ""
            if surname and surname in ml_by_surname:
                candidates = ml_by_surname[surname]
                if len(candidates) == 1:
                    # Single candidate with matching surname — accept it
                    matched = candidates[0]
                    match_by_name += 1
                    log.debug(
                        "Surname match for %s → %s",
                        player.get("player_name"),
                        candidates[0].get("player_name"),
                    )
                elif len(candidates) > 1 and pteam:
                    # Multiple candidates — disambiguate by team
                    for c in candidates:
                        cand_team = _normalise(
                            str(c.get("teamName", c.get("team_name", "")))
                        )
                        if pteam and cand_team and pteam == cand_team:
                            matched = c
                            match_by_name += 1
                            break

        if matched is not None:
            player["has_ml_data"] = True
            player["predicted_fantavoto"] = matched.get("predicted") or matched.get(
                "predicted_fantavoto"
            )
            player["prediction_std"] = matched.get("prediction_std")
            player["expected_minutes"] = matched.get("expected_minutes") or matched.get(
                "expectedMinutes"
            )

            # Enrich with VAR data
            resolved_id = pid if pid is not None else matched.get("player_fotmob_id")
            if resolved_id is not None and int(resolved_id) in var_by_id:
                v = var_by_id[int(resolved_id)]
                player["var_score"] = v.get("var_score")
                player["esv"] = v.get("esv")

            # Enrich with next-season prediction
            ns = ns_by_name.get(pname)
            if ns is not None:
                player["next_season_predicted"] = ns.get(
                    "predictedNextFantavoto"
                ) or ns.get("predicted_next_fantavoto")
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
        "Merge complete: %d by fotmob_id, %d by bridge, %d by name, %d without ML data",
        match_by_id,
        match_by_bridge,
        match_by_name,
        no_ml,
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
