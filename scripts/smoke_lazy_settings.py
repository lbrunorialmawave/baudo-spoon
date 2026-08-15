"""Smoke test: verifica che MLConfig e i call site R2-only funzionino senza ML_DATABASE_URL."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml.config import MLConfig, settings, ARTIFACTS_DIR, BASE_DIR  # noqa: E402

print("[OK] import ml.config (singleton) riuscito senza ML_DATABASE_URL")
print(f"[OK] settings.r2_bucket_name = {settings.r2_bucket_name!r}")
db_state = "set" if settings.database_url else "<unset (None)>"
print(f"[OK] settings.database_url {db_state}")
try:
    settings.get_database_url()
    print("[FAIL] Atteso RuntimeError, non sollevato")
    sys.exit(1)
except RuntimeError as e:
    print(f"[OK] get_database_url() -> RuntimeError atteso: {e}")

# Verifica import rollout (catena che in origine causava l'errore)
from ml.run_rollout import _build_parser  # noqa: E402
print("[OK] import ml.run_rollout riuscito (la catena MLConfig->canary non blocca)")
parser = _build_parser()
print(f"[OK] _build_parser() -> ArgumentParser con subcommand: {[s for s in parser._actions if hasattr(s, 'choices') and s.choices]}")

# Con env impostata
import os
os.environ["ML_DATABASE_URL"] = "postgresql+psycopg2://test:5432/x"
cfg = MLConfig()
print(f"[OK] cfg.get_database_url() = {cfg.get_database_url()!r}")
