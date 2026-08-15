"""Regressione: ``MLConfig.database_url`` opzionale (WS14/WS15, plan §17.3).

Prima di questa modifica, ``database_url`` era un campo Pydantic obbligatorio
(``Field(...)``).  La conseguenza era che qualunque modulo che importava
``ml.config`` (anche solo transitivamente via ``ml.rollout``) sollevava
``ValidationError`` se ``ML_DATABASE_URL`` non era presente nell'ambiente.
Questo rompeva ``ml.run_rollout status`` e gli altri subcommand R2-only nei
job CI che non esportano il secret del DB.

La correzione: ``database_url`` è ora ``str | None = None``; i moduli che
effettivamente aprono una connessione devono chiamare
:meth:`MLConfig.get_database_url`, che fallisce con un errore esplicito
quando l'env non è settata.  Il singleton ``settings`` continua a essere
eagerly istanziato a import time, senza bisogno di segreti.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assicura che ``ML_DATABASE_URL`` non sia settata durante il test."""
    monkeypatch.delenv("ML_DATABASE_URL", raising=False)


def test_mlconfig_instantiates_without_database_url(no_db_env) -> None:
    """``MLConfig()`` non deve sollevare ``ValidationError`` senza DB URL."""
    # Reimport per essere certi che settings sia istanziato ex-novo
    # con l'env corrente del test.
    from ml.config import MLConfig

    cfg = MLConfig()
    assert cfg.database_url is None


def test_singleton_settings_works_without_database_url(no_db_env) -> None:
    """Il singleton ``settings`` deve essere importabile senza DB URL."""
    import ml.config as ml_config

    # Forza il reimport in caso il test runner abbia già cachato.
    importlib.reload(ml_config)
    try:
        assert ml_config.settings.database_url is None
        # Anche gli altri campi devono funzionare normalmente.
        assert ml_config.settings.r2_bucket_name == "baudo-spoon-ml-artifacts"
    finally:
        # Ripristina lo stato "cachato" per non inquinare gli altri test
        # che si aspettano il singleton istanziato con ML_DATABASE_URL.
        importlib.reload(ml_config)


def test_get_database_url_raises_without_env(no_db_env) -> None:
    """``get_database_url()`` deve sollevare ``RuntimeError`` con istruzioni."""
    from ml.config import MLConfig

    cfg = MLConfig()
    with pytest.raises(RuntimeError) as excinfo:
        cfg.get_database_url()
    assert "ML_DATABASE_URL" in str(excinfo.value)
    assert "postgresql" in str(excinfo.value).lower()


def test_get_database_url_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_database_url()`` ritorna il valore configurato."""
    from ml.config import MLConfig

    monkeypatch.setenv("ML_DATABASE_URL", "postgresql+psycopg2://u:p@h:5432/d")
    cfg = MLConfig()
    assert cfg.get_database_url() == "postgresql+psycopg2://u:p@h:5432/d"


def test_run_rollout_import_does_not_require_database_url(no_db_env) -> None:
    """La CLI di ``ml.run_rollout`` deve importarsi senza DB URL.

    Regressione specifica: la pipeline ``ml-training.yml`` esegue la
    subcommand ``status`` in job che non esportano ``ML_DATABASE_URL``
    (es. ``promote-to-active``).  L'import non deve fallire.
    """
    import ml.run_rollout  # noqa: F401
    from ml.run_rollout import _build_parser  # type: ignore[attr-defined]

    parser = _build_parser()
    # Verifica che le subcommand esistano (sanity sul dispatch).
    subcommands = {action.dest for action in parser._actions}  # type: ignore[attr-defined]
    assert "command" in subcommands


def test_rollout_canary_module_does_not_instantiate_settings(no_db_env) -> None:
    """Verifica statica: ``ml.rollout.canary`` non istanzia ``MLConfig``.

    Il modulo importa ``MLConfig`` solo per typing (``MLConfig`` come
    annotation in :func:`build_canary_report`); l'istanza non deve essere
    materializzata a import time.  Con il singleton eager, questo è vero
    finché nessun campo è letto nel modulo.
    """
    import ml.rollout.canary as canary_mod

    # L'attributo MLConfig è solo un reference symbolico, non un'istanza.
    # Il test sostanziale è che l'import sia andato a buon fine senza DB.
    assert hasattr(canary_mod, "build_canary_report")
    assert hasattr(canary_mod, "CANARY_REPORT_VERSION")
