#!/usr/bin/env python
"""Fantasy-football ML offline experiments — command-line entry point.

Esegue la matrice di varianti A/B/C/D (vedi ``ml.experiments.harness``) e
produce un report comparativo (``report.json`` + sottocartelle per
variante).  Il report è l'unico input per la decisione di rollout
descritta in PR8 del piano ``plan.md``.

Usage examples
--------------
# Matrice canonica completa (A + B + C + D):

    ML_DATABASE_URL="postgresql+psycopg2://fbref:pass@localhost:5432/fbref" \\
    python -m ml.run_experiments

# Solo control + weighting (A + B) per smoke-test rapido:

    python -m ml.run_experiments --variants A_control,B_weighting

# Variante custom con fantavoto reale e prior di shrinkage più conservativo:

    python -m ml.run_experiments \\
        --fantavoto-csv path/to/fantavoto.csv \\
        --shrinkage-prior-strength 500

# Output in directory dedicata + log JSON per ELK/Splunk:

    python -m ml.run_experiments --output-dir /tmp/exp --json-logs

# Docker (legge ML_DATABASE_URL dalle env, montate da docker-compose):

    docker compose run --rm api python -m ml.run_experiments

Exit codes
----------
0 — esperimento completato (anche se una variante è fallita; controlla
    ``report.json`` → ``variants[*].status``)
1 — errore fatale (DB irraggiungibile, eccezione non gestita)
2 — errore di configurazione (env mancanti, CSV inesistente)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Structured JSON logging (riusa il pattern di run_pipeline.py) ─────────────

class JsonFormatter(logging.Formatter):
    """Formattatore JSON single-line, identico a ``ml.run_pipeline``.

    Mantenuto come copia locale (anziché importato) per evitare
    accoppiamenti e per consentire l'uso di ``run_experiments`` come
    stand-alone senza dover inizializzare l'intero package ``ml``.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_obj["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(log_obj, ensure_ascii=False)


def _configure_logging(level: str, json_logs: bool = False) -> None:
    """Configura il logger root con formato JSON o human-readable."""
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ml.run_experiments",
        description=(
            "Esegue la matrice di varianti offline (PR5) e produce un "
            "report comparativo per guidare la decisione di rollout (PR8)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--league",
        default=None,
        metavar="NAME",
        help=(
            "Filtra per lega (match parziale, es. 'Serie A'). "
            "Default = MLConfig.league_name. Imposta ML_LEAGUE_NAME='' per "
            "optare per training multi-lega."
        ),
    )
    parser.add_argument(
        "--fantavoto-csv",
        default=None,
        metavar="PATH",
        dest="fantavoto_csv",
        help=(
            "Path al CSV con fantavoto_medio reali. Se omesso, il target "
            "viene approssimato dalle statistiche FotMob."
        ),
    )
    parser.add_argument(
        "--test-seasons",
        type=int,
        default=1,
        dest="test_seasons",
        metavar="N",
        help="Numero di stagioni più recenti da tenere fuori dal train.",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=800,
        dest="min_minutes",
        metavar="N",
        help="Minuti minimi per inclusione nel cohort STANDARD.",
    )
    parser.add_argument(
        "--min-minutes-hard",
        type=int,
        default=100,
        dest="min_minutes_hard",
        metavar="N",
        help=(
            "Cutoff duro per il cohort LIMITED (PR1). Sotto questa soglia "
            "il giocatore resta escluso anche quando la sperimentazione "
            "low-sample è attiva."
        ),
    )
    parser.add_argument(
        "--weighting-strategy",
        default="sqrt",
        dest="weighting_strategy",
        choices=["constant", "linear", "sqrt", "bucketed"],
        help=(
            "Strategia di sample weighting per righe LIMITED "
            "(vedi ml.sample_reliability.weights)."
        ),
    )
    parser.add_argument(
        "--shrinkage-prior-strength",
        type=int,
        default=300,
        dest="shrinkage_prior_strength",
        metavar="N",
        help=(
            "Peso del prior nella shrinkage bayesiana per-90 (PR3). "
            "Più alto = più regolarizzazione verso la media di popolazione."
        ),
    )
    parser.add_argument(
        "--variants",
        default="A_control,B_weighting,C_shrinkage,D_recent_role_features",
        metavar="LIST",
        help=(
            "Lista CSV delle varianti da eseguire, scelte fra: "
            "A_control, B_weighting, C_shrinkage, D_recent_role_features. "
            "Usa 'all' per la matrice completa (default)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        metavar="DIR",
        help=(
            "Directory radice per gli artefatti. Default: "
            "<artifacts_dir>/experiments/<run_id>/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed per riproducibilità.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity del logging.",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        dest="json_logs",
        help="Emetti log come JSON single-line (per ELK/Splunk).",
    )
    return parser.parse_args()


def _resolve_variants(spec: str) -> list[str]:
    """Parsa la stringa ``--variants`` e ritorna la lista risolta.

    Supporta l'alias ``all`` per indicare l'intera matrice canonica.
    Le stringhe vuote o i nomi non riconosciuti sollevano ``SystemExit``
    con un messaggio esplicito verso l'utente.
    """
    spec_norm = (spec or "").strip()
    if not spec_norm or spec_norm.lower() == "all":
        # Matrice canonica completa.
        return [
            "A_control",
            "B_weighting",
            "C_shrinkage",
            "D_recent_role_features",
        ]
    requested = [s.strip() for s in spec_norm.split(",") if s.strip()]
    known = {
        "A_control",
        "B_weighting",
        "C_shrinkage",
        "D_recent_role_features",
    }
    unknown = [r for r in requested if r not in known]
    if unknown:
        raise SystemExit(
            f"Varianti non riconosciute: {unknown}. "
            f"Scegli fra: {sorted(known)} oppure 'all'."
        )
    return requested


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level, json_logs=args.json_logs)
    log = logging.getLogger(__name__)

    # ── Validazione database URL ─────────────────────────────────────────────
    db_url = os.environ.get("ML_DATABASE_URL") or os.environ.get("API_DATABASE_URL")
    if not db_url:
        log.error(
            "Database URL non configurato. "
            "Imposta la variabile d'ambiente ML_DATABASE_URL."
        )
        return 2

    # ── Validazione fantavoto CSV ────────────────────────────────────────────
    fantavoto_csv = Path(args.fantavoto_csv) if args.fantavoto_csv else None
    if fantavoto_csv and not fantavoto_csv.exists():
        log.error("fantavoto CSV non trovato: %s", fantavoto_csv)
        return 2

    # ── Risoluzione varianti (prima dell'import lazy) ────────────────────────
    try:
        selected = _resolve_variants(args.variants)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        log.exception("Errore nel parsing di --variants")
        return 2

    # ── Setup env per pydantic-settings (prima dell'import lazy) ────────────
    os.environ["ML_DATABASE_URL"] = db_url
    os.environ["ML_LOG_LEVEL"] = args.log_level
    os.environ["ML_RANDOM_SEED"] = str(args.seed)
    os.environ["ML_TEST_SEASONS"] = str(args.test_seasons)
    os.environ["ML_MIN_MINUTES"] = str(args.min_minutes)
    os.environ["ML_MIN_MINUTES_HARD"] = str(args.min_minutes_hard)
    os.environ["ML_WEIGHTING_STRATEGY"] = args.weighting_strategy
    os.environ["ML_SHRINKAGE_PRIOR_STRENGTH"] = str(args.shrinkage_prior_strength)
    if args.league:
        os.environ["ML_LEAGUE_NAME"] = args.league
    if args.output_dir:
        os.environ["ML_ARTIFACTS_DIR"] = args.output_dir

    # ── Import lazy (dopo setup env) ────────────────────────────────────────
    from ml.config import MLConfig
    from ml.experiments import (
        VARIANT_A,
        VARIANT_B,
        VARIANT_C,
        VARIANT_D,
        default_variants,
        run_experiment,
    )

    cfg = MLConfig()
    _configure_logging(cfg.log_level, json_logs=args.json_logs)

    # ── Filtra la matrice canonica in base a --variants ─────────────────────
    all_variants = default_variants()
    selected_dict = {name: all_variants[name] for name in selected}
    log.info(
        "Avvio esperimento con %d variante/i: %s",
        len(selected_dict),
        list(selected_dict.keys()),
    )
    log.debug(
        "Config base: min_minutes=%d, test_seasons=%d, weighting=%s",
        cfg.min_minutes,
        cfg.test_seasons,
        cfg.weighting_strategy,
    )

    # ── Esecuzione ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        report = run_experiment(
            cfg,
            variants=selected_dict,
            external_fantavoto_csv=fantavoto_csv,
            output_dir=output_dir,
        )
    except Exception:  # noqa: BLE001
        log.exception("Esecuzione esperimento fallita.")
        return 1

    # ── Sommario finale (JSON-line su stdout per l'action) ──────────────────
    summary = {
        "run_id": report.get("run_id"),
        "variants_executed": list(report.get("variants", {}).keys()),
        "variants_failed": [
            name
            for name, payload in report.get("variants", {}).items()
            if payload.get("status") == "error"
        ],
        "report_path": str(
            (output_dir or (cfg.artifacts_dir / "experiments" / report["run_id"]))
            / "report.json"
        ),
    }
    # Pretty summary con confronto side-by-side
    for name, payload in report.get("variants", {}).items():
        if payload.get("status") != "ok":
            summary.setdefault("details", {})[name] = {"status": "error"}
            continue
        summary.setdefault("details", {})[name] = {
            "best_model": payload.get("best_model"),
            "rmse": payload.get("rmse"),
            "mae": payload.get("mae"),
            "r2": payload.get("r2"),
            "backtest_mean_rmse": payload.get("backtest_mean_rmse"),
        }

    print(json.dumps(summary, indent=2, default=str))

    # Exit code 0 anche se una variante è fallita: l'utente deve leggere il
    # sommario.  Errori fatali (eccezioni non gestite) escono con 1 sopra.
    return 0


if __name__ == "__main__":
    sys.exit(main())
