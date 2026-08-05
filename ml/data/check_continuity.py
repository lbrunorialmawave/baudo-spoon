"""Standalone season-continuity check — reuses ``load_raw_data``'s existing
gap-detection logic (see ``ml/data/loader.py``, "SEASON GAP DETECTED" warning)
without running the full training pipeline. Safe to run at any time,
including before the new season's data has landed.

Usage: python -m ml.data.check_continuity
"""

from __future__ import annotations

import logging

import sqlalchemy as sa

from ..config import MLConfig
from .loader import load_raw_data


def main() -> int:
    cfg = MLConfig()
    logging.basicConfig(level=cfg.log_level)
    engine = sa.create_engine(cfg.database_url)
    load_raw_data(engine, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
