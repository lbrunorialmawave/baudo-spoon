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
    DEPTH_ORDER,
    POOL_FUSIONE,
    calcola_ruolo_primario,
    calcola_pool_esteso,
    ALL_ROLES,
)

__all__ = [
    "MantraConfig",
    "PROFONDITA_MAP",
    "POOL_FUSIONE",
    "calcola_ruolo_primario",
    "calcola_pool_esteso",
    "DEPTH_ORDER",
    "ALL_ROLES",
]
