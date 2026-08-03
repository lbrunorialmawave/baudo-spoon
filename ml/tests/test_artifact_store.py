"""Unit test per :mod:`ml.storage.artifact_store`.

boto3 è mockato manualmente (nessuna dipendenza da moto, non presente nei
requirements di ml/): copre upload ok/fallito, download ok/fallito, e i
quattro stati di `exists()` (LOCAL, REMOTE_ONLY, MISSING, R2_UNREACHABLE).

Questo è anche il test di regressione diretto per il bug che ha originato
il piano "R2 come source of truth" (vedi test_exists_remote_only_when_r2_populated_and_local_empty).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.storage.artifact_store import ArtifactAvailability, ArtifactStore, R2Config


class _ClientError(Exception):
    """Stand-in leggero per botocore.exceptions.ClientError."""

    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


@pytest.fixture
def r2_config() -> R2Config:
    return R2Config(
        endpoint_url="https://fake-account.r2.cloudflarestorage.com",
        access_key_id="fake-key",
        secret_access_key="fake-secret",
        bucket_name="test-bucket",
    )


@pytest.fixture
def store(tmp_path: Path, r2_config: R2Config) -> ArtifactStore:
    s = ArtifactStore(local_dir=tmp_path, r2_config=r2_config)
    s._s3 = MagicMock()  # evita di istanziare un vero client boto3
    return s


# ── save_json / upload ──────────────────────────────────────────────────────


def test_save_json_writes_local_file(store: ArtifactStore, tmp_path: Path) -> None:
    import json

    path = store.save_json({"a": 1}, "foo.json")
    assert path == tmp_path / "foo.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_save_json_calls_r2_upload(store: ArtifactStore) -> None:
    store.save_json({"a": 1}, "foo.json")
    store._s3.upload_file.assert_called_once()
    args, kwargs = store._s3.upload_file.call_args
    assert args[1] == "test-bucket"
    assert args[2] == "foo.json"


def test_save_json_upload_failure_is_non_fatal(store: ArtifactStore, tmp_path: Path) -> None:
    store._s3.upload_file.side_effect = RuntimeError("expired credentials")
    # Non deve sollevare — il file locale deve comunque esistere.
    path = store.save_json({"a": 1}, "foo.json")
    assert path.exists()


def test_save_json_without_r2_config_skips_upload(tmp_path: Path) -> None:
    s = ArtifactStore(local_dir=tmp_path, r2_config=None)
    path = s.save_json({"a": 1}, "foo.json")
    assert path.exists()  # nessun crash, nessun tentativo di upload


def test_save_json_no_tmp_file_left_behind_on_write_error(store: ArtifactStore, tmp_path: Path, monkeypatch) -> None:
    import json as json_module

    def _boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(json_module, "dump", _boom)
    with pytest.raises(ValueError):
        store.save_json({"a": 1}, "foo.json")
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


# ── save_binary ──────────────────────────────────────────────────────────────


def test_save_binary_copies_and_uploads(store: ArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "model_src.bin"
    src.write_bytes(b"fake-model-bytes")
    dest = store.save_binary(src, "model_latest.bin")
    assert dest == tmp_path / "model_latest.bin"
    assert dest.read_bytes() == b"fake-model-bytes"
    store._s3.upload_file.assert_called_once()


# ── load_json ────────────────────────────────────────────────────────────────


def test_load_json_reads_local_without_touching_r2(store: ArtifactStore, tmp_path: Path) -> None:
    (tmp_path / "foo.json").write_text('{"a": 1}', encoding="utf-8")
    result = store.load_json("foo.json")
    assert result == {"a": 1}
    store._s3.download_file.assert_not_called()


def test_load_json_falls_back_to_r2_when_local_missing(store: ArtifactStore, tmp_path: Path) -> None:
    def _fake_download(bucket, key, dest):
        Path(dest).write_text('{"b": 2}', encoding="utf-8")

    store._s3.download_file.side_effect = _fake_download
    result = store.load_json("foo.json")
    assert result == {"b": 2}
    store._s3.download_file.assert_called_once_with("test-bucket", "foo.json", str(tmp_path / "foo.json"))


def test_load_json_returns_none_when_missing_everywhere(store: ArtifactStore) -> None:
    store._s3.download_file.side_effect = _ClientError("404")
    assert store.load_json("nope.json") is None


def test_load_json_returns_none_when_r2_unreachable(store: ArtifactStore) -> None:
    store._s3.download_file.side_effect = RuntimeError("connection timed out")
    assert store.load_json("foo.json") is None


def test_load_json_without_r2_config_returns_none_when_local_missing(tmp_path: Path) -> None:
    s = ArtifactStore(local_dir=tmp_path, r2_config=None)
    assert s.load_json("nope.json") is None


# ── exists(): i quattro stati ────────────────────────────────────────────────


def test_exists_local(store: ArtifactStore, tmp_path: Path) -> None:
    (tmp_path / "foo.json").write_text("{}", encoding="utf-8")
    assert store.exists("foo.json") == ArtifactAvailability.LOCAL
    store._s3.head_object.assert_not_called()  # local hit non deve toccare R2


def test_exists_remote_only_when_r2_populated_and_local_empty(store: ArtifactStore) -> None:
    """Test di regressione diretto per il bug originario: readiness endpoint
    su un'istanza con disco locale vuoto ma artefatto presente su R2 deve
    risultare disponibile, non 'missing'.
    """
    store._s3.head_object.return_value = {"ContentLength": 123}
    assert store.exists("results_latest.json") == ArtifactAvailability.REMOTE_ONLY


def test_exists_missing_when_absent_both_local_and_r2(store: ArtifactStore) -> None:
    store._s3.head_object.side_effect = _ClientError("404")
    assert store.exists("nope.json") == ArtifactAvailability.MISSING


def test_exists_r2_unreachable_distinct_from_missing(store: ArtifactStore) -> None:
    """Un problema di credenziali/rete non deve mai travestirsi da 'missing'."""
    store._s3.head_object.side_effect = RuntimeError("could not connect to endpoint")
    assert store.exists("foo.json") == ArtifactAvailability.R2_UNREACHABLE


def test_exists_missing_when_no_r2_config_and_local_absent(tmp_path: Path) -> None:
    s = ArtifactStore(local_dir=tmp_path, r2_config=None)
    assert s.exists("nope.json") == ArtifactAvailability.MISSING


# ── R2Config.from_env() ──────────────────────────────────────────────────────
# Regressione: costruire R2Config a partire dal singleton MLConfig/Settings
# obbliga a fornire anche campi non correlati come `database_url`
# (obbligatorio, senza default) — from_env() legge solo i 4 campi R2_*
# direttamente dall'ambiente, così i chiamanti che non hanno già in mano una
# Settings interamente popolata (es. ml.mantra.runner.run_mantra, chiamato
# con engine=None nei test) non acquisiscono quella dipendenza indiretta.


def test_r2_config_from_env_reads_prefixed_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_R2_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("ML_R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("ML_R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("ML_R2_BUCKET_NAME", "my-bucket")
    cfg = R2Config.from_env()
    assert cfg == R2Config(
        endpoint_url="https://acct.r2.cloudflarestorage.com",
        access_key_id="key",
        secret_access_key="secret",
        bucket_name="my-bucket",
    )


def test_r2_config_from_env_does_not_require_unrelated_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nessuna variabile R2_* impostata (e nessun ML_DATABASE_URL in giro,
    a differenza di quanto avviene se si passa per il singleton MLConfig):
    from_env() deve comunque ritornare senza sollevare eccezioni.
    """
    for key in ("ML_R2_ENDPOINT_URL", "ML_R2_ACCESS_KEY_ID", "ML_R2_SECRET_ACCESS_KEY", "ML_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    cfg = R2Config.from_env()
    assert cfg.is_configured is False
    assert cfg.bucket_name == "baudo-spoon-ml-artifacts"


# ── R2Config.is_configured ───────────────────────────────────────────────────


def test_r2_config_not_configured_without_endpoint() -> None:
    cfg = R2Config(endpoint_url=None, access_key_id=None, secret_access_key=None, bucket_name="b")
    assert cfg.is_configured is False


def test_store_with_unconfigured_r2_behaves_as_local_only(tmp_path: Path) -> None:
    cfg = R2Config(endpoint_url=None, access_key_id=None, secret_access_key=None, bucket_name="b")
    s = ArtifactStore(local_dir=tmp_path, r2_config=cfg)
    assert s._r2_config is None  # normalizzato a None dal costruttore
    path = s.save_json({"x": 1}, "foo.json")
    assert path.exists()


# ── wrapper async ─────────────────────────────────────────────────────────────


def test_load_json_async_delegates_to_sync(store: ArtifactStore, tmp_path: Path) -> None:
    (tmp_path / "foo.json").write_text('{"a": 1}', encoding="utf-8")
    result = asyncio.run(store.load_json_async("foo.json"))
    assert result == {"a": 1}


def test_exists_async_delegates_to_sync(store: ArtifactStore, tmp_path: Path) -> None:
    (tmp_path / "foo.json").write_text("{}", encoding="utf-8")
    result = asyncio.run(store.exists_async("foo.json"))
    assert result == ArtifactAvailability.LOCAL
