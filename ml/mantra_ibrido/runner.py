"""Orchestrator: load MANTRA + ML → merge → score → classify → persist.

``run_hybrid_computation`` is the single entry point for the whole hybrid
pipeline.  It accepts an optional ``output_filename`` parameter that allows
callers to write to a **preview** file instead of the production artefact,
keeping experimental results invisible to regular users.
"""

from __future__ import annotations

import logging
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
    # Optional ID-map file bridges fantacalcio_id → player_fotmob_id
    id_map_path = output_dir / "player_id_map.json"
    merged = merge_datasets(mantra_path, ml_path, id_map_path=id_map_path)

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

    # ── Persist (atomic write + best-effort R2 upload — see design doc
    # "R2 come source of truth", Fase 2) ────────────────────────────────────
    output_dir = Path(output_dir)
    filename = output_filename or f"mantra_ibrido_results_{season}.json"

    from ml.storage.artifact_store import ArtifactStore, R2Config

    store = ArtifactStore(local_dir=output_dir, r2_config=R2Config.from_env())
    final_path = store.save_json(result, filename)

    log.info("Hybrid results written to %s", final_path)
    return result
