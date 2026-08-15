#!/usr/bin/env python3
"""Detect configuration drift tra rollout dichiarato e effective_config (Phase 7).

Questo modulo sostituisce lo script inline presente in
``.github/workflows/ml-training.yml`` (Phase 7 — Validate config drift).

Vantaggi rispetto allo script inline
------------------------------------

* Evita l'errore ``ml.run_pipeline: error: unrecognized arguments: python -``
  che si verificava passando ``python -`` come argomento al container
  ``ml-runner:latest`` (il cui ``ENTRYPOINT`` è ``python -m ml.run_pipeline``).
* Rende la logica testabile localmente (``python -m ml.scripts.detect_config_drift``).
* Usa l'API pubblica di :mod:`ml.rollout.config_drift` senza
  passare parametri non supportati (es. ``rollout_pct`` / ``flag``) che
  causerebbero ``TypeError`` a runtime.
* Scrive il report in ``artifacts/config_drift.json`` (path
  indipendente dal CWD del container) e ritorna l'exit code canonico
  ``1`` in caso di P0, ``0`` altrimenti.

Exit codes
----------

* ``0`` — nessun drift P0 (pipeline prosegue)
* ``1`` — almeno un finding ``P0`` (FAIL-closed, blocca la pipeline)
* ``2`` — errore I/O o file mancante

Configurazione
--------------

Lo script legge i seguenti percorsi:

* ``artifacts/rollout/state.json``  — stato R2 (sempre presente dopo Phase 1)
* ``${ARTIFACT_EFFECTIVE_CONFIG}``  — effective_config.json (default:
  ``artifacts/effective_config.json``)
* ``${PROMOTE_FLAG}``               — opzionale, seleziona il flag specifico
  per il quale costruire lo snapshot. Se vuoto, viene usato il primo flag
  dichiarato in ordine alfabetico; se non ci sono flag, lo snapshot
  viene costruito con stage ``DISABLED``.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Any

from ml.rollout.config_drift import (
    DriftReport,
    EffectiveConfig,
    FlagStage,
    RolloutSnapshot,
    detect_config_drift,
    effective_config_from_mapping,
)
from ml.rollout.controller import reliability_weight_mode_for_stage

log = logging.getLogger("ml.scripts.detect_config_drift")
if not log.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


# ── Costruzione snapshot da file di stato rollout ──────────────────────────


_DEFAULT_ROLLOUT_STATE = pathlib.Path("artifacts/rollout/state.json")
_DEFAULT_EFFECTIVE_CONFIG = pathlib.Path("artifacts/effective_config.json")


def _load_rollout_state(path: pathlib.Path) -> dict[str, Any]:
    """Carica ``state.json`` oppure ritorna un fallback DISABLED.

    Il fallback corrisponde a quello già usato da Phase 1 quando lo stato
    non è ancora stato pubblicato in R2.
    """
    if not path.is_file():
        log.warning("rollout state %s non trovato, uso fallback DISABLED", path)
        return {"version": 1, "flags": {}, "audit": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _select_target_flag(flags: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Seleziona il flag target in base a ``PROMOTE_FLAG`` o al primo presente.

    Returns:
        (flag_name, flag_state). Se ``flags`` è vuoto ritorna
        ``("", {"stage": "disabled", "rollout_pct": 0.0})`` per
        garantire uno snapshot valido.
    """
    if not flags:
        return "", {"stage": "disabled", "rollout_pct": 0.0}

    target = os.environ.get("PROMOTE_FLAG", "").strip()
    if target and target in flags:
        return target, flags[target]

    # Default deterministico: primo flag in ordine alfabetico.  Stessa
    # euristica dello script inline originale (Phase 7 prima del fix).
    first_key = sorted(flags.keys())[0]
    return first_key, flags[first_key]


def _coerce_stage(raw: Any) -> FlagStage:
    """Coerce uno stage \"raw\" nel corrispondente membro di :class:`FlagStage`."""
    if isinstance(raw, FlagStage):
        return raw
    value = str(raw or "disabled").strip().lower()
    try:
        return FlagStage(value)
    except ValueError:
        log.warning(
            "stage %r non riconosciuto, fallback a DISABLED", raw
        )
        return FlagStage.DISABLED


def build_rollout_snapshot(
    rollout_state: dict[str, Any],
) -> tuple[RolloutSnapshot, str]:
    """Costruisce un :class:`RolloutSnapshot` dal contenuto di ``state.json``.

    Returns:
        (snapshot, target_flag). ``target_flag`` può essere la stringa
        vuota se non è stato dichiarato alcun flag.
    """
    flags = rollout_state.get("flags") or {}
    target, fs = _select_target_flag(flags)
    stage = _coerce_stage(fs.get("stage", "disabled"))
    return (
        RolloutSnapshot(
            stage=stage,
            production_mode=reliability_weight_mode_for_stage(stage),
            production_flags={},
            challenger_flags={},
            source=f"rollout_state[{target or 'none'}]",
        ),
        target,
    )


# ── Main ──────────────────────────────────────────────────────────────────


def _resolve_effective_path() -> pathlib.Path:
    raw = os.environ.get("ARTIFACT_EFFECTIVE_CONFIG", "").strip()
    if raw:
        return pathlib.Path(raw)
    return _DEFAULT_EFFECTIVE_CONFIG


def detect(
    *,
    rollout_state_path: pathlib.Path | str = _DEFAULT_ROLLOUT_STATE,
    effective_config_path: pathlib.Path | str | None = None,
    output_path: pathlib.Path | str | None = None,
) -> DriftReport:
    """Esegue il detect e scrive il report.

    Args:
        rollout_state_path: path a ``rollout/state.json``.
        effective_config_path: path a ``effective_config.json``. Se ``None``
            usa ``$ARTIFACT_EFFECTIVE_CONFIG`` o il default.
        output_path: path del file JSON di output. Se ``None`` usa
            ``artifacts/config_drift.json`` (relativo al CWD).

    Returns:
        :class:`DriftReport`.
    """
    rollout_path = pathlib.Path(rollout_state_path)
    effective_path = pathlib.Path(
        effective_config_path
        if effective_config_path is not None
        else _resolve_effective_path()
    )
    out_path = pathlib.Path(
        output_path if output_path is not None else "artifacts/config_drift.json"
    )

    if not effective_path.is_file():
        log.error("effective_config non trovato: %s", effective_path)
        raise FileNotFoundError(f"effective_config not found: {effective_path}")

    rollout_state = _load_rollout_state(rollout_path)
    snapshot, target = build_rollout_snapshot(rollout_state)
    effective_data = json.loads(effective_path.read_text(encoding="utf-8"))
    # Mantieni il flag in `source` per tracciabilità, anche se la dataclass
    # non lo prevede come campo dedicato.
    effective_source = (
        f"effective_config[{target or 'none'}]"
        if target
        else "effective_config"
    )
    effective = effective_config_from_mapping(effective_data, source=effective_source)

    log.info(
        "detect_config_drift: target=%s stage=%s effective_mode=%s",
        target or "<none>",
        snapshot.stage.value,
        effective.production_mode,
    )

    report = detect_config_drift(snapshot, effective)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json(indent=2), encoding="utf-8")

    print("=== config drift report ===")
    print(report.to_json(indent=2))
    print(
        f"drift exit_code={report.exit_code()} "
        f"highest_severity={report.highest_severity.value if report.highest_severity else 'NONE'}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ritorna 0/1 per CI, 2 per errori I/O."""
    try:
        report = detect()
    except FileNotFoundError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"::error::config drift detection failed: {exc}", file=sys.stderr)
        return 2
    return int(report.exit_code())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
