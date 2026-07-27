"""Orchestrator: load MANTRA + ML → merge → score → classify → persist.

``run_hybrid_computation`` is the single entry point for the whole hybrid
pipeline.  It accepts an optional ``output_filename`` parameter that allows
callers to write to a **preview** file instead of the production artefact,
keeping experimental results invisible to regular users.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .classifications import compute_hybrid_classifications
from .config import MantraIbridoConfig
from .config_store import load_config
from .merger import merge_datasets
from .scoring import compute_hybrid_scores

log = logging.getLogger(__name__)


def run_hybrid_computation(
    mantra_path: Path,
    ml_path: Path,
    output_dir: Path,
    config: MantraIbridoConfig | None = None,
    output_filename: str | None = None,
) -> dict[str, Any]:
    """Execute the full hybrid MANTRA+ML pipeline.

    Parameters
    ----------
    mantra_path:
        Path to ``mantra_results_{season}.json``.
    ml_path:
        Path to ``results_latest.json``.
    output_dir:
        Directory where the result artefact will be written.
    config:
        Optional override.  When ``None``, the persisted config is loaded
        via :func:`~config_store.load_config`.
    output_filename:
        When provided, the result is written under this filename **instead of**
        the canonical ``mantra_ibrido_results_{season}.json``.  Used by the
        admin preview feature so experimental runs never overwrite production
        data.

    Returns
    -------
    dict with keys ``meta``, ``players``, ``classifications`` — ready to be
    serialised as JSON and served by the API.

    Raises
    ------
    ValueError
        If the MANTRA artefact does not contain ``meta.season_start``.
    """
    if config is None:
        config = load_config()

    log.info("=" * 60)
    log.info("Hybrid computation — weights: MANTRA=%.2f / ML=%.2f",
             config.PESO_MANTRA, config.PESO_ML)

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = merge_datasets(mantra_path, ml_path)

    season = merged["meta"].get("season_start")
    if season is None:
        raise ValueError(
            "Impossibile determinare la stagione: meta.season_start mancante "
            "nel file MANTRA"
        )

    # ── Score ─────────────────────────────────────────────────────────────────
    players_ibridi = compute_hybrid_scores(merged["players"], config)
    log.info("Scored %d players (hybrid)", len(players_ibridi))

    # ── Classify ──────────────────────────────────────────────────────────────
    classifications = compute_hybrid_classifications(players_ibridi, config)

    # Attach hybridLabels to each player (reverse lookup from classification lists)
    label_to_players: dict[str, set[str]] = {}
    for label_name, player_names in classifications.items():
        label_to_players[label_name] = set(player_names)

    for p in players_ibridi:
        name = str(p.get("player_name", ""))
        p["hybridLabels"] = [
            label for label, names in label_to_players.items()
            if name in names
        ]

    # ── Assemble output ──────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "meta": {
            "seasonStart": season,
            "generatedAt": merged["meta"].get("generated_at"),
            "config": asdict(config),
            "nPlayersMantra": merged["meta"].get("n_players_mantra"),
            "nPlayersWithMl": merged["meta"].get("n_players_with_ml"),
            "nPlayersWithoutMl": merged["meta"].get("n_players_without_ml"),
            "mantraRunId": merged["meta"].get("mantra_run_id"),
            "mlRunId": merged["meta"].get("ml_run_id"),
        },
        "players": players_ibridi,
        "classifications": classifications,
    }

    # ── Persist (atomic write) ────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_filename or f"mantra_ibrido_results_{season}.json"
    final_path = output_dir / filename

    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    log.info("Hybrid results written to %s", final_path)
    return result
