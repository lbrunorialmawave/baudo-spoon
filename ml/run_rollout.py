#!/usr/bin/env python
"""Production rollout controller CLI (PR8 of the low-sample plan).

Gestisce i feature flag del piano low-sample (``ml.rollout.controller``)
da riga di comando, con persistenza su filesystem locale + sync opzionale
su Cloudflare R2.  È l'unico entry point che il workflow
``.github/workflows/ml-rollout.yml`` invoca per promuovere/demote flag.

Subcommands
-----------
status
    Stampa lo stage corrente di ogni flag + ultime N entry di audit.

shadow <FLAG> [--rollout-pct PCT] [--note TEXT] [--actor NAME]
    Promuove un flag a ``SHADOW`` (dual scoring, no impact in produzione).

activate <FLAG> [--rollout-pct PCT] [--note TEXT] [--actor NAME]
    Promuove un flag a ``ACTIVE`` (autoritativo per la quota ``rollout_pct``).

disable <FLAG> [--note TEXT] [--actor NAME]
    Torna a ``DISABLED`` (rollback di emergenza).

audit [--limit N]
    Dumpa l'audit log completo (o le ultime N entry).

save-snapshot --name NAME [--commit-sha SHA] [--actor NAME]
    Salva uno snapshot dello stato corrente (known-good config).

list-snapshots
    Elenca gli snapshot salvati (più recenti prima).

restore-snapshot --name NAME [--reason TEXT] [--actor NAME] [--trigger TEXT]
    Ripristina lo stato da uno snapshot (rollback mirato).

rollback-all [--reason TEXT] [--actor NAME] [--trigger TEXT]
    Forza tutti i flag a ``DISABLED`` (kill switch di emergenza, WS17).

Usage examples
--------------
# Stato corrente (tutti i flag, ultimi 10 eventi):
    python -m ml.run_rollout status

# Promuove shrinkage in SHADOW al 10%:
    python -m ml.run_rollout shadow enable_shrinkage \\
        --rollout-pct 10 --note "Post PR5 exp 20260812" --actor lbrunori

# Dopo un monitoraggio positivo, attiva:
    python -m ml.run_rollout activate enable_shrinkage --rollout-pct 100

# Rollback di emergenza:
    python -m ml.run_rollout disable enable_shrinkage \\
        --note "Anomalia metriche shadow" --actor lbrunori

# Log audit completo:
    python -m ml.run_rollout audit --limit 50

# WS17 — snapshot pre-deploy:
    python -m ml.run_rollout save-snapshot --name pre-shrinkage-2026-08-12

# WS17 — rollback totale di emergenza (kill switch):
    python -m ml.run_rollout rollback-all \\
        --reason "canary anomaly in top decile" --actor lbrunori

# WS17 — ripristino a snapshot noto:
    python -m ml.run_rollout restore-snapshot --name pre-shrinkage-2026-08-12 \\
        --reason "anomaly correlata alla nuova promozione" --actor lbrunori

# Da GitHub Actions (con R2 sync):
    python -m ml.run_rollout activate enable_shrinkage \\
        --r2-bucket baudo-spoon-ml-artifacts --sync-r2

Exit codes
----------
0 — successo
1 — errore fatale (eccezione non gestita, R2 sync fallita se richiesta)
2 — errore di configurazione (env mancanti, flag sconosciuto, stage invalido)
3 — promotion gate negato (fail-closed)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


# ── Structured JSON logging (identico a run_pipeline.py / run_experiments.py) ─


class JsonFormatter(logging.Formatter):
    """Formattatore JSON single-line per ELK/Splunk."""

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


# ── Persistenza stato rollout ────────────────────────────────────────────────

STATE_FILENAME: Final[str] = "state.json"
STATE_VERSION: Final[int] = 1
# Prefix R2: rollout/state.json e rollout/audit.json
R2_STATE_KEY: Final[str] = "rollout/state.json"
R2_AUDIT_KEY: Final[str] = "rollout/audit.json"
LOCAL_ROLLOUT_SUBDIR: Final[str] = "rollout"


@dataclass
class FlagState:
    """Stato corrente di un singolo feature flag.

    Attributes:
        flag: Nome canonico del flag (corrisponde a ``MLConfig`` field).
        stage: Stage corrente (``disabled`` / ``shadow`` / ``active``).
        rollout_pct: Quota di traffico in ``ACTIVE`` (0-100).
        updated_at: Timestamp ISO-8601 UTC dell'ultima transizione.
        updated_by: Attore che ha eseguito l'ultima transizione.
        note: Nota opzionale dell'operatore.
    """

    flag: str
    stage: str = "disabled"
    rollout_pct: float = 0.0
    updated_at: str = ""
    updated_by: str = ""
    note: str = ""


@dataclass
class RolloutState:
    """Stato globale dei feature flag + audit log.

    Attributes:
        version: Schema version (per forward-compat).
        updated_at: Timestamp dell'ultima modifica.
        flags: Mappa flag → :class:`FlagState`.
        audit: Lista di eventi (append-only).
    """

    version: int = STATE_VERSION
    updated_at: str = ""
    flags: dict[str, FlagState] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "flags": {k: vars(v) for k, v in self.flags.items()},
            "audit": list(self.audit),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RolloutState":
        flags: dict[str, FlagState] = {}
        for name, raw in (payload.get("flags") or {}).items():
            flags[name] = FlagState(
                flag=name,
                stage=raw.get("stage", "disabled"),
                rollout_pct=float(raw.get("rollout_pct", 0.0)),
                updated_at=raw.get("updated_at", ""),
                updated_by=raw.get("updated_by", ""),
                note=raw.get("note", ""),
            )
        return cls(
            version=int(payload.get("version", STATE_VERSION)),
            updated_at=payload.get("updated_at", ""),
            flags=flags,
            audit=list(payload.get("audit") or []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    @classmethod
    def empty(cls) -> "RolloutState":
        """Stato iniziale: tutti i flag noti a ``DISABLED``."""
        # Import lazy per evitare problemi di import circolari.
        from ml.rollout import FeatureFlag

        flags = {
            flag.value: FlagState(flag=flag.value) for flag in FeatureFlag
        }
        return cls(flags=flags)


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _local_state_path(base: Path) -> Path:
    return base / LOCAL_ROLLOUT_SUBDIR / STATE_FILENAME


def load_state(local_path: Path) -> RolloutState:
    """Carica lo stato da file locale.  Se assente o malformato, ritorna vuoto."""
    if not local_path.exists():
        log.info("Nessuno stato locale trovato in %s → stato vuoto.", local_path)
        return RolloutState.empty()
    try:
        payload = json.loads(local_path.read_text(encoding="utf-8"))
        state = RolloutState.from_dict(payload)
        log.info(
            "Stato locale caricato: %d flag, %d eventi audit.",
            len(state.flags), len(state.audit),
        )
        return state
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        log.warning(
            "File di stato corrotto in %s: %s → stato vuoto.",
            local_path, exc,
        )
        return RolloutState.empty()


def save_state(state: RolloutState, local_path: Path) -> None:
    """Salva lo stato localmente (atomic write)."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(tz=timezone.utc).isoformat()
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    tmp.write_text(state.to_json(), encoding="utf-8")
    tmp.replace(local_path)
    log.info("Stato salvato in %s.", local_path)


# ── R2 sync (opzionale) ──────────────────────────────────────────────────────


def _r2_client():
    """Build a boto3 S3 client pointed at Cloudflare R2.

    Usa le env ``ML_R2_*`` (come nel resto del progetto). Fallback alle
    classiche ``AWS_*`` per compatibilità locale.
    """
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("ML_R2_ENDPOINT_URL", "")
    if not endpoint:
        return None

    access_key = os.environ.get("ML_R2_ACCESS_KEY_ID") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    )
    secret_key = os.environ.get("ML_R2_SECRET_ACCESS_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )


def _download_from_r2(key: str, bucket: str, dest: Path) -> bool:
    """Scarica ``key`` da R2 via boto3.  Ritorna True se scaricato."""
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)

    client = _r2_client()
    if client is None:
        log.warning("ML_R2_ENDPOINT_URL non configurato, skip download R2.")
        return False

    log.info("Download R2: s3://%s/%s → %s", bucket, key, dest)
    try:
        client.download_file(bucket, key, str(dest))
    except client.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            log.info("File R2 %s non presente (prima esecuzione?).", key)
            return False
        log.error("Download R2 fallito: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        log.error("Download R2 fallito: %s", exc)
        return False

    return dest.exists() and dest.stat().st_size > 0


def _upload_to_r2(local: Path, key: str, bucket: str) -> bool:
    """Carica ``local`` su R2 via boto3.  Ritorna True se ok."""
    if not local.exists():
        log.error("File locale %s assente, upload R2 saltato.", local)
        return False

    client = _r2_client()
    if client is None:
        log.warning("ML_R2_ENDPOINT_URL non configurato, skip upload R2.")
        return False

    log.info("Upload R2: %s → s3://%s/%s", local, bucket, key)
    try:
        client.upload_file(str(local), bucket, key)
    except Exception as exc:  # noqa: BLE001
        log.error("Upload R2 fallito: %s", exc)
        return False
    return True


# ── Azioni di promozione ─────────────────────────────────────────────────────


def _validate_flag_name(name: str) -> str:
    """Verifica che il flag sia un ``FeatureFlag`` valido.  Ritorna il valore."""
    from ml.rollout import FeatureFlag

    try:
        return FeatureFlag(name).value
    except ValueError:
        valid = ", ".join(f.value for f in FeatureFlag)
        raise SystemExit(
            f"Flag '{name}' non riconosciuto. Valori validi: {valid}"
        )


def _apply_transition(
    state: RolloutState,
    flag_value: str,
    new_stage: str,
    rollout_pct: float,
    actor: str,
    note: str,
    *,
    promotion_report: Path | None = None,
    config_snapshot: Path | None = None,
    break_glass: bool = False,
    break_glass_reason: str | None = None,
) -> dict[str, Any]:
    """Applica una transizione di stage al flag e registra l'evento.

    Quando ``new_stage == "active"`` e ``promotion_report`` è fornito,
    la transizione passa attraverso
    :meth:`RolloutController.promote_to_active` (WS6 — fail-closed
    promotion gate).  In assenza di report, la transizione usa il
    percorso storico :meth:`RolloutController.promote` (compat).
    """
    from ml.rollout import FeatureFlag, FlagStage, RolloutController

    # Mappa il flag value al enum per usare la logica del controller.
    flag_enum = FeatureFlag(flag_value)
    current = state.flags.get(
        flag_value,
        FlagState(flag=flag_value),
    )
    controller = RolloutController(
        flag=flag_enum,
        stage=FlagStage(current.stage),
        rollout_pct=current.rollout_pct,
    )

    # Calcola "from" PRIMA di promuovere (per l'audit).
    from_stage = current.stage
    from_pct = current.rollout_pct

    # Esegui la promozione.
    gate_outcome: dict[str, Any] | None = None
    if new_stage == "active" and promotion_report is not None:
        # Hard promotion gate (WS6).  Solleva PromotionGateDenied
        # in caso di fail-closed.  Con break_glass=True il gate
        # viene override-ato ma il motivo viene registrato.
        from ml.rollout import (
            PromotionGateDenied,
            PromotionGateError,
        )

        try:
            outcome = controller.promote_to_active(
                report_path=promotion_report,
                config_snapshot_path=config_snapshot,
                actor=actor,
                new_rollout_pct=rollout_pct,
                break_glass=break_glass,
                break_glass_reason=break_glass_reason,
            )
        except PromotionGateDenied as exc:
            print(
                json.dumps(
                    {
                        "error": "promotion_denied",
                        "message": str(exc),
                        "outcome": exc.outcome.to_dict(),
                    }
                ),
                file=sys.stderr,
            )
            raise
        except PromotionGateError as exc:
            print(
                json.dumps(
                    {
                        "error": "promotion_gate_error",
                        "message": str(exc),
                    }
                ),
                file=sys.stderr,
            )
            raise

        # Log strutturato dell'esito del gate (per audit CI).
        gate_outcome = {
            "passed": outcome.passed,
            "failures": list(outcome.failures),
            "config_hash": outcome.config_hash,
            "config_hash_status": outcome.config_hash_status,
            "report_path": outcome.report_path,
            "break_glass": break_glass,
        }
        log.info(
            "Promotion gate evaluated for %s: passed=%s failures=%d",
            flag_value, outcome.passed, len(outcome.failures),
        )
    else:
        # Percorso storico: la logica monotonica è nel controller.
        controller.promote(
            new_stage=FlagStage(new_stage),
            new_rollout_pct=rollout_pct,
        )

    # Aggiorna lo stato persistito.
    new_state = FlagState(
        flag=flag_value,
        stage=controller.stage.value,
        rollout_pct=controller.rollout_pct,
        updated_at=datetime.now(tz=timezone.utc).isoformat(),
        updated_by=actor,
        note=note,
    )
    state.flags[flag_value] = new_state

    # Aggiungi l'evento audit prendendolo dal controller.
    last_event = controller.events[-1]
    last_event["actor"] = actor
    last_event["note"] = note
    last_event["from_stage"] = from_stage
    last_event["from_pct"] = from_pct
    if gate_outcome is not None:
        last_event["gate_result"] = "PASS" if gate_outcome["passed"] else "BREAK_GLASS"
        last_event["gate_failures"] = gate_outcome["failures"]
        last_event["gate_config_hash"] = gate_outcome["config_hash"]
        last_event["gate_config_hash_status"] = gate_outcome["config_hash_status"]
        last_event["promotion_report"] = gate_outcome["report_path"]
    state.audit.append(last_event)

    return {
        "flag": flag_value,
        "from_stage": from_stage,
        "to_stage": controller.stage.value,
        "rollout_pct": controller.rollout_pct,
        "actor": actor,
        "note": note,
        "at": new_state.updated_at,
    }


# ── Subcommands ──────────────────────────────────────────────────────────────


def cmd_status(state: RolloutState, args: argparse.Namespace) -> int:
    """Stampa lo stato corrente di tutti i flag + ultime N entry audit."""
    summary = {
        "flags": {
            name: {
                "stage": fs.stage,
                "rollout_pct": fs.rollout_pct,
                "updated_at": fs.updated_at,
                "updated_by": fs.updated_by,
                "note": fs.note,
            }
            for name, fs in state.flags.items()
        },
        "recent_audit": state.audit[-args.limit:],
        "total_audit_events": len(state.audit),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_transition(
    state: RolloutState,
    args: argparse.Namespace,
    new_stage: str,
) -> int:
    """Logica condivisa per shadow/activate/disable."""
    if args.flag is None:
        print(
            json.dumps({"error": f"Subcommand '{args.command}' richiede --flag"}),
            file=sys.stderr,
        )
        return 2

    flag_value = _validate_flag_name(args.flag)
    # `disable` non espone --rollout-pct; default sicuro a 0.0.
    rollout_pct = getattr(args, "rollout_pct", 0.0) or 0.0

    if new_stage == "active" and rollout_pct <= 0.0:
        print(
            json.dumps(
                {"error": "Promozione ad ACTIVE richiede --rollout-pct > 0"}
            ),
            file=sys.stderr,
        )
        return 2

    if not 0.0 <= rollout_pct <= 100.0:
        print(
            json.dumps(
                {"error": f"--rollout-pct deve essere in [0, 100], ricevuto {rollout_pct}"}
            ),
            file=sys.stderr,
        )
        return 2

    # Break-glass richiede --break-glass-reason non vuoto.
    break_glass = bool(getattr(args, "break_glass", False))
    break_glass_reason = getattr(args, "break_glass_reason", None)
    if break_glass and not (break_glass_reason and str(break_glass_reason).strip()):
        print(
            json.dumps(
                {
                    "error": (
                        "--break-glass richiede --break-glass-reason non vuoto"
                    ),
                }
            ),
            file=sys.stderr,
        )
        return 2

    try:
        event = _apply_transition(
            state=state,
            flag_value=flag_value,
            new_stage=new_stage,
            rollout_pct=rollout_pct,
            actor=args.actor,
            note=args.note or "",
            promotion_report=getattr(args, "promotion_report", None),
            config_snapshot=getattr(args, "config_snapshot", None),
            break_glass=break_glass,
            break_glass_reason=break_glass_reason,
        )
    except Exception:
        # _apply_transition ha già scritto un JSON diagnostico su stderr.
        # Restituiamo exit 3 per distinguerlo da errori CLI generici.
        return 3

    # Persisti subito (così se l'utente vede errore R2, ha già lo stato locale).
    local_path = _local_state_path(args.artifacts_dir)
    save_state(state, local_path)

    # Sync R2 opzionale.
    if args.sync_r2:
        if not _upload_to_r2(local_path, R2_STATE_KEY, args.r2_bucket):
            log.error("Sync R2 fallita.")
            return 1

    summary = {
        "status": "ok",
        "transition": event,
        "state_path": str(local_path),
        "r2_synced": args.sync_r2,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_audit(state: RolloutState, args: argparse.Namespace) -> int:
    """Dump dell'audit log."""
    limit = args.limit if args.limit and args.limit > 0 else len(state.audit)
    events = state.audit[-limit:]
    summary = {
        "total_events": len(state.audit),
        "returned": len(events),
        "events": events,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


# ── Subcommand WS17 — snapshot + rollback production-grade ─────────────────


def _flags_view_for_snapshot(
    state: RolloutState,
) -> dict[str, dict[str, Any]]:
    """Restituisce la vista ``{flag: {stage, rollout_pct}}`` dello stato corrente."""
    return {
        name: {"stage": fs.stage, "rollout_pct": fs.rollout_pct}
        for name, fs in state.flags.items()
    }


def _ensure_state_on_disk(state: RolloutState, artifacts_dir: Path) -> Path:
    """Forza la persistenza dello stato e ritorna il path del file locale."""
    local_path = _local_state_path(artifacts_dir)
    save_state(state, local_path)
    return local_path


def cmd_save_snapshot(state: RolloutState, args: argparse.Namespace) -> int:
    """Salva uno snapshot dello stato corrente (known-good config)."""
    from ml.rollout import (
        SnapshotError,
        save_snapshot as _save_snapshot,
    )

    if not args.name or not str(args.name).strip():
        print(
            json.dumps({"error": "--name obbligatorio per save-snapshot"}),
            file=sys.stderr,
        )
        return 2

    flags_view = _flags_view_for_snapshot(state)
    if not flags_view:
        print(
            json.dumps({"error": "stato vuoto: nessun flag da snapshotare"}),
            file=sys.stderr,
        )
        return 2

    try:
        snapshot = _save_snapshot(
            artifacts_root=args.artifacts_dir,
            name=str(args.name).strip(),
            flags=flags_view,
            saved_by=str(args.actor).strip() or "cli",
            commit_sha=str(args.commit_sha).strip() if args.commit_sha else None,
        )
    except SnapshotError as exc:
        print(json.dumps({"error": "snapshot_error", "message": str(exc)}), file=sys.stderr)
        return 2

    summary = {
        "status": "ok",
        "snapshot": snapshot.to_dict(),
        "path": str(snapshot_path := _snapshot_dir(args.artifacts_dir) / f"{snapshot.name}.json"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _snapshot_dir(artifacts_root: Path) -> Path:
    from ml.rollout import snapshot_dir as _snapshot_dir_fn
    return _snapshot_dir_fn(artifacts_root)


def cmd_list_snapshots(args: argparse.Namespace) -> int:
    """Elenca gli snapshot salvati (più recenti prima)."""
    from ml.rollout import list_snapshots as _list_snapshots

    snapshots = _list_snapshots(args.artifacts_dir)
    summary = {
        "total": len(snapshots),
        "snapshots": [
            {
                "name": s.name,
                "saved_at": s.saved_at,
                "saved_by": s.saved_by,
                "commit_sha": s.commit_sha,
                "config_hash": s.config_hash,
                "flags": {k: dict(v) for k, v in s.flags.items()},
            }
            for s in snapshots
        ],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _ensure_audit_log(state: RolloutState) -> "AuditLog":  # type: ignore[name-defined]
    """Costruisce un :class:`AuditLog` a partire dall'audit in-memory dello stato.

    L'audit log è in-memory per il rollback: una volta persistito viene
    riversato nello stato e nel file di log come eventi ``rollback``.
    """
    from ml.rollout import AuditLog, records_from_controller_events

    audit = AuditLog()
    if state.audit:
        # Trasforma gli eventi del controller in record audit, mantenendo
        # eventuali record espliciti (record_transition / record_denied / record_rollback)
        # che sono già nella forma ``dict`` con chiave ``kind``.
        for ev in state.audit:
            if isinstance(ev, dict) and "kind" in ev and "actor" in ev:
                # Record serializzato in precedenza: lo reincludiamo solo se
                # ha la forma attesa (per evitare di duplicare eventi del
                # controller come se fossero record di audit).
                if ev.get("kind") in {"transition", "denied", "rollback"}:
                    from ml.rollout import (
                        AuditKind,
                        AuditRecord,
                    )
                    audit.append(
                        AuditRecord(
                            kind=AuditKind(ev["kind"]),
                            timestamp=str(ev.get("timestamp", ev.get("at", ""))),
                            actor=str(ev.get("actor", "unknown")),
                            flag=ev.get("flag"),
                            from_stage=ev.get("from_stage"),
                            to_stage=ev.get("to_stage"),
                            from_pct=ev.get("from_pct"),
                            to_pct=ev.get("to_pct"),
                            reason=str(ev.get("reason", "?")),
                            commit_sha=ev.get("commit_sha"),
                            promotion_report=ev.get("promotion_report"),
                            gate_result=ev.get("gate_result"),
                            config_hash=ev.get("config_hash"),
                            failed_checks=tuple(ev.get("failed_checks", ())),
                            extra=dict(ev.get("extra", {})),
                        )
                    )
        # Aggiunge anche i record derivati dagli eventi "grezzi" del
        # controller che NON sono già nel formato audit-record.
        raw_events = [ev for ev in state.audit if not (isinstance(ev, dict) and ev.get("kind") in {"transition", "denied", "rollback"})]
        if raw_events:
            audit.extend(records_from_controller_events(raw_events, actor="state-replay"))
    return audit


def _commit_sha_from_state(state: RolloutState) -> str | None:
    """Estrae il commit SHA più recente dallo stato, se presente."""
    for entry in reversed(state.audit):
        if not isinstance(entry, dict):
            continue
        sha = entry.get("commit_sha")
        if sha:
            return str(sha)
    return None


def cmd_rollback_all(state: RolloutState, args: argparse.Namespace) -> int:
    """Forza tutti i flag noti a ``DISABLED`` (kill switch, WS17)."""
    from ml.rollout import (
        AuditLog,
        rollback_all_to_disabled,
    )

    if not args.reason or not str(args.reason).strip():
        print(
            json.dumps({"error": "--reason obbligatorio per rollback-all"}),
            file=sys.stderr,
        )
        return 2

    audit: AuditLog = _ensure_audit_log(state)
    state_flags_view = _flags_view_for_snapshot(state)
    commit_sha = (
        str(args.commit_sha).strip()
        if args.commit_sha
        else _commit_sha_from_state(state)
    )
    new_flags, report = rollback_all_to_disabled(
        state_flags=state_flags_view,
        audit_log=audit,
        actor=str(args.actor).strip() or "cli",
        reason=str(args.reason).strip(),
        commit_sha=commit_sha,
        trigger=str(args.trigger).strip() if args.trigger else "manual",
    )

    # Aggiorna lo stato persistito: tutti i flag noti sono a DISABLED/0.
    now = datetime.now(tz=timezone.utc).isoformat()
    for name, fs in state.flags.items():
        if name in new_flags:
            new_state = new_flags[name]
            state.flags[name] = FlagState(
                flag=name,
                stage=str(new_state.get("stage", "disabled")),
                rollout_pct=float(new_state.get("rollout_pct", 0.0)),
                updated_at=str(new_state.get("updated_at", now)),
                updated_by=str(new_state.get("updated_by", args.actor)),
                note=str(args.reason).strip(),
            )
    # Aggiungi gli eventi rollback-only all'audit dello stato.
    for record in audit.records:
        if record.kind.value == "rollback":
            state.audit.append(record.to_dict())

    local_path = _ensure_state_on_disk(state, args.artifacts_dir)
    if args.sync_r2:
        if not _upload_to_r2(local_path, R2_STATE_KEY, args.r2_bucket):
            log.error("Sync R2 fallita.")
            return 1

    summary = {
        "status": "ok",
        "report": report.to_dict(),
        "state_path": str(local_path),
        "r2_synced": args.sync_r2,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_restore_snapshot(state: RolloutState, args: argparse.Namespace) -> int:
    """Ripristina lo stato da uno snapshot (WS17)."""
    from ml.rollout import (
        AuditLog,
        SnapshotError,
        load_snapshot as _load_snapshot,
        rollback_to_snapshot,
    )

    if not args.name or not str(args.name).strip():
        print(
            json.dumps({"error": "--name obbligatorio per restore-snapshot"}),
            file=sys.stderr,
        )
        return 2
    if not args.reason or not str(args.reason).strip():
        print(
            json.dumps({"error": "--reason obbligatorio per restore-snapshot"}),
            file=sys.stderr,
        )
        return 2

    try:
        snapshot = _load_snapshot(args.artifacts_dir, str(args.name).strip())
    except SnapshotError as exc:
        print(json.dumps({"error": "snapshot_error", "message": str(exc)}), file=sys.stderr)
        return 2

    audit: AuditLog = _ensure_audit_log(state)
    state_flags_view = _flags_view_for_snapshot(state)
    commit_sha = (
        str(args.commit_sha).strip()
        if args.commit_sha
        else _commit_sha_from_state(state)
    )
    new_flags, report = rollback_to_snapshot(
        state_flags=state_flags_view,
        audit_log=audit,
        snapshot=snapshot,
        actor=str(args.actor).strip() or "cli",
        reason=str(args.reason).strip(),
        commit_sha=commit_sha,
        trigger=str(args.trigger).strip() if args.trigger else "manual",
    )

    now = datetime.now(tz=timezone.utc).isoformat()
    for name, fs in state.flags.items():
        if name in new_flags:
            new_state = new_flags[name]
            state.flags[name] = FlagState(
                flag=name,
                stage=str(new_state.get("stage", "disabled")),
                rollout_pct=float(new_state.get("rollout_pct", 0.0)),
                updated_at=str(new_state.get("updated_at", now)),
                updated_by=str(new_state.get("updated_by", args.actor)),
                note=str(args.reason).strip(),
            )
    for record in audit.records:
        if record.kind.value == "rollback":
            state.audit.append(record.to_dict())

    local_path = _ensure_state_on_disk(state, args.artifacts_dir)
    if args.sync_r2:
        if not _upload_to_r2(local_path, R2_STATE_KEY, args.r2_bucket):
            log.error("Sync R2 fallita.")
            return 1

    summary = {
        "status": "ok",
        "report": report.to_dict(),
        "state_path": str(local_path),
        "r2_synced": args.sync_r2,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


# ── CLI parser ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.run_rollout",
        description=(
            "Controller CLI per i feature flag low-sample (PR8). "
            "Gestisce promozione a SHADOW/ACTIVE, rollback, audit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        metavar="DIR",
        help="Directory radice per il file di stato locale.",
    )
    parser.add_argument(
        "--r2-bucket",
        default=os.environ.get("ML_R2_BUCKET_NAME", "baudo-spoon-ml-artifacts"),
        metavar="NAME",
        help="Bucket R2 dove sincronizzare lo stato (se --sync-r2).",
    )
    parser.add_argument(
        "--sync-r2",
        action="store_true",
        help="Scarica stato da R2 prima, ricarica dopo la transizione.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity del logging.",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emetti log come JSON single-line.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = sub.add_parser("status", help="Stato corrente di tutti i flag.")
    p_status.add_argument(
        "--limit", type=int, default=10,
        help="Quante entry di audit recenti includere.",
    )

    # shadow
    p_shadow = sub.add_parser("shadow", help="Promuove un flag a SHADOW.")
    p_shadow.add_argument("flag", nargs="?", default=None, help="Nome del flag.")
    p_shadow.add_argument(
        "--rollout-pct", type=float, default=0.0,
        help="Quota di traffico (0-100). In SHADOW non viene usata ma registrata.",
    )
    p_shadow.add_argument("--actor", default="cli", help="Chi esegue l'azione.")
    p_shadow.add_argument("--note", default="", help="Nota opzionale.")

    # activate
    p_activate = sub.add_parser("activate", help="Promuove un flag a ACTIVE.")
    p_activate.add_argument("flag", nargs="?", default=None, help="Nome del flag.")
    p_activate.add_argument(
        "--rollout-pct", type=float, default=100.0,
        help="Quota di traffico (0-100). Default=100 = tutti.",
    )
    p_activate.add_argument("--actor", default="cli", help="Chi esegue l'azione.")
    p_activate.add_argument("--note", default="", help="Nota opzionale.")
    p_activate.add_argument(
        "--promotion-report",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path al report JSON dell'experiment harness.  Se fornito, "
            "la transizione ad ACTIVE passa attraverso il promotion gate "
            "fail-closed (plan §8).  Se il gate non passa, la transizione "
            "viene negata (exit 3) salvo --break-glass."
        ),
    )
    p_activate.add_argument(
        "--config-snapshot",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path al JSON di config snapshot.  Se fornito insieme a "
            "--promotion-report, il config_hash del candidate viene "
            "confrontato con quello registrato nel report (plan §18).  "
            "Mismatch forza DENY (mai WARN)."
        ),
    )
    p_activate.add_argument(
        "--break-glass",
        action="store_true",
        help=(
            "Override di emergenza del promotion gate.  Richiede "
            "--break-glass-reason.  MAI usare nel workflow standard.  "
            "Disponibile solo in ml-rollout.yml con input esplicito."
        ),
    )
    p_activate.add_argument(
        "--break-glass-reason",
        default=None,
        metavar="TEXT",
        help="Motivazione dell'override (obbligatoria con --break-glass).",
    )

    # disable
    p_disable = sub.add_parser("disable", help="Torna a DISABLED (rollback).")
    p_disable.add_argument("flag", nargs="?", default=None, help="Nome del flag.")
    p_disable.add_argument("--actor", default="cli", help="Chi esegue l'azione.")
    p_disable.add_argument("--note", default="", help="Motivo del rollback.")

    # audit
    p_audit = sub.add_parser("audit", help="Dump audit log.")
    p_audit.add_argument(
        "--limit", type=int, default=50,
        help="Quante entry ritornare (0 = tutte).",
    )

    # WS17 — save-snapshot
    p_save_snap = sub.add_parser(
        "save-snapshot",
        help="Salva uno snapshot dello stato corrente (known-good config).",
    )
    p_save_snap.add_argument(
        "--name", required=True, metavar="NAME",
        help=(
            "Identificatore dello snapshot. Usare uno schema tipo "
            "'pre-<flag>-<YYYYMMDD>' per garantire l'ordinamento. "
            "Caratteri ammessi: [A-Za-z0-9._-], lunghezza 1-128."
        ),
    )
    p_save_snap.add_argument(
        "--actor", default="cli", help="Operatore o sistema che cattura lo snapshot.",
    )
    p_save_snap.add_argument(
        "--commit-sha", default=None, metavar="SHA",
        help="Commit SHA da associare allo snapshot (per audit).",
    )

    # WS17 — list-snapshots
    sub.add_parser(
        "list-snapshots",
        help="Elenca gli snapshot salvati (più recenti prima).",
    )

    # WS17 — restore-snapshot
    p_restore = sub.add_parser(
        "restore-snapshot",
        help="Ripristina lo stato da uno snapshot (rollback mirato, WS17).",
    )
    p_restore.add_argument(
        "--name", required=True, metavar="NAME",
        help="Nome dello snapshot da ripristinare.",
    )
    p_restore.add_argument(
        "--reason", required=True, metavar="TEXT",
        help="Motivo del rollback (obbligatorio per audit).",
    )
    p_restore.add_argument(
        "--actor", default="cli", help="Operatore o sistema che esegue il rollback.",
    )
    p_restore.add_argument(
        "--trigger", default="manual", metavar="TRIGGER",
        help=(
            "Identificatore del trigger (default: manual). Valori tipici: "
            "promotion_regression, canary_anomaly, config_drift, "
            "runtime_error_threshold, invariant_violation."
        ),
    )
    p_restore.add_argument(
        "--commit-sha", default=None, metavar="SHA",
        help="Commit SHA al momento del rollback (opzionale).",
    )

    # WS17 — rollback-all
    p_rb_all = sub.add_parser(
        "rollback-all",
        help="Forza tutti i flag a DISABLED (kill switch, WS17).",
    )
    p_rb_all.add_argument(
        "--reason", required=True, metavar="TEXT",
        help="Motivo del rollback totale (obbligatorio per audit).",
    )
    p_rb_all.add_argument(
        "--actor", default="cli", help="Operatore o sistema che esegue il rollback.",
    )
    p_rb_all.add_argument(
        "--trigger", default="manual", metavar="TRIGGER",
        help=(
            "Identificatore del trigger (default: manual). Valori tipici: "
            "promotion_regression, canary_anomaly, config_drift, "
            "runtime_error_threshold, invariant_violation."
        ),
    )
    p_rb_all.add_argument(
        "--commit-sha", default=None, metavar="SHA",
        help="Commit SHA al momento del rollback (opzionale).",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level, json_logs=args.json_logs)
    global log
    log = logging.getLogger(__name__)

    # Carica stato iniziale: prima da R2 (se richiesto), poi da locale, poi vuoto.
    state: RolloutState = RolloutState.empty()
    local_path = _local_state_path(args.artifacts_dir)

    if args.sync_r2:
        # Scarica in un file temporaneo e poi prova a leggerlo.
        tmp_path = local_path.with_suffix(".r2tmp")
        if _download_from_r2(R2_STATE_KEY, args.r2_bucket, tmp_path):
            try:
                payload = json.loads(tmp_path.read_text(encoding="utf-8"))
                state = RolloutState.from_dict(payload)
                log.info("Stato caricato da R2 (%d flag).", len(state.flags))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("Stato R2 corrotto: %s → uso locale.", exc)
                state = load_state(local_path)
        else:
            state = load_state(local_path)
    else:
        state = load_state(local_path)

    # Dispatch subcommand.
    if args.command == "status":
        return cmd_status(state, args)
    if args.command == "shadow":
        return cmd_transition(state, args, new_stage="shadow")
    if args.command == "activate":
        return cmd_transition(state, args, new_stage="active")
    if args.command == "disable":
        return cmd_transition(state, args, new_stage="disabled")
    if args.command == "audit":
        return cmd_audit(state, args)
    if args.command == "save-snapshot":
        return cmd_save_snapshot(state, args)
    if args.command == "list-snapshots":
        return cmd_list_snapshots(args)
    if args.command == "restore-snapshot":
        return cmd_restore_snapshot(state, args)
    if args.command == "rollback-all":
        return cmd_rollback_all(state, args)

    parser.error(f"Comando sconosciuto: {args.command}")
    return 2  # unreachable


# Logger globale (impostato in main()).
log = logging.getLogger(__name__)


if __name__ == "__main__":
    sys.exit(main())