"""Pytest bootstrap: rende ``api`` importabile in locale e in Docker.

In locale (Windows) il backend FastAPI vive in ``api/src/``; in Docker
il ``WORKDIR`` è ``/app`` e ``src/`` viene rinominato in radice.  Per
rendere i test agnostici rispetto a dove viene eseguita la suite,
mettiamo sia la root del progetto sia la root backend nel ``sys.path``
e creiamo un alias ``api.routers`` → ``api.src.routers`` per non
dover toccare i test esistenti.

Le variabili d'ambiente minime per inizializzare ``APISettings`` senza
un vero database sono impostate qui: i test del router ``/auction`` non
toccano la connessione al DB perché usano solo il ``player_pool`` del
payload.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Root del progetto (= due livelli sopra questo file: api/tests/conftest.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "api" / "src"

for path in (str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Env minima: pydantic-settings richiede ``API_DATABASE_URL`` per
# inizializzare ``APISettings``. I test del router ``/auction`` non
# aprono connessioni reali, ma l'import del modulo lo richiede.
os.environ.setdefault(
    "API_DATABASE_URL",
    "postgresql+psycopg://test:test@localhost:5432/test",
)
# NOTA: API_API_KEY_SECRET lasciato vuoto di proposito: ``verify_api_key``
# skippa la validazione quando ``settings.api_key_secret`` è vuoto
# (modalità dev).  Se in futuro vorremo testare l'auth end-to-end,
# basta valorizzare la env e aggiungere l'header ``X-API-Key`` al
# TestClient.

# Alias: ``api.routers`` deve risolvere a ``api.src.routers``.
# Garantisce che i test esistenti (scritti pensando a ``api`` come root)
# funzionino anche con la struttura ``api/src/`` del workspace locale.
try:
    _routers = importlib.import_module("api.src.routers")
    sys.modules.setdefault("api.routers", _routers)
except ModuleNotFoundError:
    # In Docker ``api`` non esiste: lo schema atteso è ``from routers``.
    pass

# Override della fixture ``client`` definita localmente nei test:
# ``TestClient`` di default solleva le eccezioni sollevate negli handler
# (``raise_server_exceptions=True``).  Il router ``/auction`` lascia
# propagare alcuni ``ValueError`` interni come 500, e diversi test si
# aspettano proprio il 500.  Disattiviamo la propagazione automatica
# delle eccezioni per ripristinare la semantica "tutto → response".
import pytest as _pytest
from fastapi.testclient import TestClient as _TestClient


@_pytest.fixture
def client() -> _TestClient:
    """Re-dichiara la fixture ``client`` con ``raise_server_exceptions=False``.

    Questo override non tocca i test esistenti: pytest usa la fixture
    più vicina al test (in questo caso, quella di questo ``conftest``).
    """
    from api.routers import auction as _auction_router  # via alias sopra

    _app = _pytest.importorskip("fastapi").FastAPI()
    _app.include_router(_auction_router.router)
    return _TestClient(_app, raise_server_exceptions=False)
