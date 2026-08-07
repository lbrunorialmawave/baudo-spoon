"""Unica porta d'ingresso per leggere/scrivere artefatti della pipeline ML.

Vedi design doc "R2 come source of truth per gli artefatti ML/MANTRA" (2026-08-02).

Pattern cache-aside:
    - ogni scrittura locale è seguita da un tentativo di upload su R2
      (best-effort, mai fatale: un fallimento R2 non deve mai interrompere
      una run della pipeline);
    - ogni lettura prova prima il disco locale, poi scarica da R2 se assente;
    - nessuna eccezione viene mai propagata al chiamante per un problema di
      R2 (credenziali scadute, bucket rinominato, rete irraggiungibile) —
      il chiamante riceve sempre un valore di ritorno esplicito e decide lui
      il comportamento (es. 503 vs "ready: false").

`exists()` distingue esplicitamente "l'artefatto non esiste ancora"
(`MISSING`) da "R2 non è raggiungibile" (`R2_UNREACHABLE`), cosa che il
vecchio codice (basato su `except: pass` o `Path.exists()` locale) non
poteva fare: un problema operativo di R2 non deve mai travestirsi da "dati
non ancora pronti".
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ArtifactAvailability(str, Enum):
    """Stato di disponibilità di un artefatto per un dato filename."""

    LOCAL = "local"  # presente sul disco locale (cache hit)
    REMOTE_ONLY = "remote_only"  # assente in locale, presente su R2
    MISSING = "missing"  # assente sia in locale che su R2 (dato non pronto)
    R2_UNREACHABLE = "r2_unreachable"  # R2 non raggiungibile — NON significa "missing"


@dataclass(frozen=True)
class R2Config:
    """Credenziali/endpoint R2. Costruita da ciascun servizio a partire dalle
    proprie Settings esistenti (`ml.config.MLConfig` / `api.src.config.Settings`).
    """

    endpoint_url: str | None
    access_key_id: str | None
    secret_access_key: str | None
    bucket_name: str

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint_url)

    @classmethod
    def from_env(cls, prefix: str = "ML_") -> R2Config:
        """Costruisce R2Config leggendo direttamente le variabili d'ambiente
        ``{prefix}R2_*``, senza passare per ``MLConfig``/``Settings``.

        Utile per moduli che non ricevono già un'istanza di Settings
        interamente popolata (es. ``ml.mantra.runner.run_mantra``, che accetta
        un ``engine`` già connesso e non richiede altrimenti configurazione
        DB) — evitare di istanziare l'intero ``MLConfig`` qui evita di
        introdurre una dipendenza indiretta da campi non correlati come
        ``database_url`` (obbligatorio, senza default).
        """
        return cls(
            endpoint_url=os.environ.get(f"{prefix}R2_ENDPOINT_URL") or None,
            access_key_id=os.environ.get(f"{prefix}R2_ACCESS_KEY_ID") or None,
            secret_access_key=os.environ.get(f"{prefix}R2_SECRET_ACCESS_KEY") or None,
            bucket_name=os.environ.get(
                f"{prefix}R2_BUCKET_NAME", "baudo-spoon-ml-artifacts"
            ),
        )


def _is_not_found_error(exc: Exception) -> bool:
    """True se l'eccezione boto3/botocore indica "chiave non trovata",
    False per qualunque altro tipo di errore (credenziali, rete, permessi).
    """
    error_code = None
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error_code = response.get("Error", {}).get("Code")
    return error_code in {"404", "NoSuchKey", "NotFound"}


class ArtifactStore:
    """Porta d'ingresso unica per lo storage degli artefatti pipeline.

    Args:
        local_dir: directory locale usata come cache L2 (sotto Redis, dove
            applicabile) e come scratch space write-side per la pipeline.
        r2_config: configurazione R2. Se ``None`` o non configurata
            (``endpoint_url`` assente), lo store opera in modalità
            "solo disco locale" — utile per test/dev senza credenziali R2.
    """

    def __init__(self, local_dir: Path, r2_config: R2Config | None) -> None:
        self._local_dir = Path(local_dir)
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._r2_config = r2_config if (r2_config and r2_config.is_configured) else None
        self._s3: Any = None

    # ── Client R2 (lazy) ─────────────────────────────────────────────────────

    def _r2_client(self) -> Any:
        if self._s3 is None:
            import boto3

            assert self._r2_config is not None
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self._r2_config.endpoint_url,
                aws_access_key_id=self._r2_config.access_key_id,
                aws_secret_access_key=self._r2_config.secret_access_key,
            )
        return self._s3

    def _local_path(self, filename: str) -> Path:
        return self._local_dir / filename

    # ── Scrittura (sincrona, usata dalla pipeline batch) ────────────────────

    def save_json(self, data: Any, filename: str) -> Path:
        """Scrive `data` come JSON sul disco locale (scrittura atomica via
        file temporaneo + rename) e tenta l'upload su R2 (best-effort).
        """
        import json

        path = self._local_path(filename)
        fd, tmp_path = tempfile.mkstemp(dir=self._local_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        log.info("artifact_store: saved local json outcome=ok key=%s", filename)
        self._upload(path, filename)
        return path

    def save_binary(self, local_src: Path, filename: str) -> Path:
        """Copia (se necessario) `local_src` nella cache locale sotto `filename`
        e tenta l'upload su R2 (best-effort). Usato per artefatti non-JSON
        (es. modelli serializzati con joblib).
        """
        local_src = Path(local_src)
        dest = self._local_path(filename)
        if local_src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_src, dest)

        log.info("artifact_store: saved local binary outcome=ok key=%s", filename)
        self._upload(dest, filename)
        return dest

    def _upload(self, path: Path, key: str) -> None:
        if self._r2_config is None:
            return
        try:
            self._r2_client().upload_file(str(path), self._r2_config.bucket_name, key)
            log.info(
                "artifact_store: upload outcome=ok operation=upload bucket=%s key=%s",
                self._r2_config.bucket_name,
                key,
            )
        except Exception as exc:
            log.warning(
                "artifact_store: upload outcome=error operation=upload bucket=%s key=%s exception=%s",
                self._r2_config.bucket_name,
                key,
                exc,
            )

    # ── Lettura (sincrona) ───────────────────────────────────────────────────

    def load_json(self, filename: str) -> dict | None:
        """Ritorna il contenuto JSON: prima dal disco locale, poi da R2
        (scaricandolo in cache locale). Ritorna ``None`` se l'artefatto non
        esiste né in locale né su R2, o se R2 non è raggiungibile — in
        entrambi i casi senza sollevare eccezioni. Per distinguere i due
        casi, usare `exists()`.
        """
        import json

        path = self._local_path(filename)
        if not path.exists():
            self._download(path, filename)

        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            log.warning(
                "artifact_store: read outcome=error operation=read_local key=%s exception=%s",
                filename,
                exc,
            )
            return None

    def _download(self, path: Path, key: str) -> None:
        if self._r2_config is None:
            return
        try:
            self._r2_client().download_file(self._r2_config.bucket_name, key, str(path))
            log.info(
                "artifact_store: download outcome=ok operation=download bucket=%s key=%s",
                self._r2_config.bucket_name,
                key,
            )
        except Exception as exc:
            outcome = "not_found" if _is_not_found_error(exc) else "r2_unreachable"
            log.warning(
                "artifact_store: download outcome=%s operation=download bucket=%s key=%s exception=%s",
                outcome,
                self._r2_config.bucket_name,
                key,
                exc,
            )
            # Rimuove eventuali file parziali lasciati da un download fallito.
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def exists(self, filename: str) -> ArtifactAvailability:
        """Determina la disponibilità di un artefatto senza scaricarlo
        (usa `head_object` su R2, non `download_file` — più economico in
        banda/tempo, adatto ai check di readiness ad alta frequenza).
        """
        if self._local_path(filename).exists():
            return ArtifactAvailability.LOCAL

        if self._r2_config is None:
            return ArtifactAvailability.MISSING

        try:
            self._r2_client().head_object(
                Bucket=self._r2_config.bucket_name, Key=filename
            )
            log.info(
                "artifact_store: exists outcome=ok operation=head bucket=%s key=%s",
                self._r2_config.bucket_name,
                filename,
            )
            return ArtifactAvailability.REMOTE_ONLY
        except Exception as exc:
            if _is_not_found_error(exc):
                log.info(
                    "artifact_store: exists outcome=not_found operation=head bucket=%s key=%s",
                    self._r2_config.bucket_name,
                    filename,
                )
                return ArtifactAvailability.MISSING
            log.warning(
                "artifact_store: exists outcome=r2_unreachable operation=head bucket=%s key=%s exception=%s",
                self._r2_config.bucket_name,
                filename,
                exc,
            )
            return ArtifactAvailability.R2_UNREACHABLE

    # ── Wrapper async (usati dall'API, thin wrapper via run_in_executor) ─────

    async def load_json_async(self, filename: str) -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load_json, filename)

    async def exists_async(self, filename: str) -> ArtifactAvailability:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.exists, filename)
