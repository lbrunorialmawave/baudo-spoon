"""MANTRA scoring engine — 12-role, 4-pillar player evaluation system.

Package structure
-----------------
config.py        — Calibratable coefficients and thresholds
roles.py         — MANTRA role definitions, depth hierarchy, pool merging
pilastro1.py     — P1: Solidità (consistency / reliability)
pilastro2.py     — P2: Potenziale (xG/xA upside)
pilastro3.py     — P3: Peso Squadra (team strength context)
pilastro4.py     — P4: Mercato Storico (historical auction value)
scoring.py       — FP → FP_Corr → VR → Prezzo_Massimo
fase7.py         — Decision rules (TOP / AFFARE / SCOMMESSA / …)
fase8.py         — Classifications (Low Cost, Watchlist, Consigliati, …)
runner.py        — Orchestrator: DB → compute → JSON

Usage
-----
    from ml.mantra.runner import run_mantra
    result = run_mantra(engine, season_start=2025)
"""

from ml.mantra.config import MantraConfig
from ml.mantra.roles import (
    ALL_ROLES,
    DEPTH_ORDER,
    POOL_FUSIONE,
    calcola_pool_esteso,
    calcola_ruolo_primario,
)

__all__ = [
    "ALL_ROLES",
    "DEPTH_ORDER",
    "POOL_FUSIONE",
    "PROFONDITA_MAP",
    "MantraConfig",
    "calcola_pool_esteso",
    "calcola_ruolo_primario",
]
